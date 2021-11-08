use std::f32::consts::{E, PI};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde_json::from_reader;
use serde::{Deserialize, Serialize};
use mfcc::Transform;
use ndarray::{Array1, Array2, Axis, concatenate, s, stack};
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, InterpreterBuilder};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PreciseError {
    #[error("Couldn't open model file")]
    FileError(#[from]std::io::Error),
    #[error("Couldn't open params file")]
    ParamsLoadError(std::io::Error),
    #[error("Params file is has bad structure or is not JSON")]
    ParamsJsonError(serde_json::Error),
    #[error("Failure while operating model")]
    TensorflowError(#[from]tflite::Error),
    #[error("Model is wrong")]
    ModelError
}

#[derive(Debug, Deserialize, Clone, Serialize)]
struct PreciseParams {
    #[serde(default = "buffer_t_default")]
    buffer_t: f32,
    #[serde(default = "window_t_default")]
    window_t: f32,
    #[serde(default = "hop_t_default")]
    hop_t: f32,
    #[serde(default = "sample_rate_default")]
    sample_rate: u16,
    #[serde(default = "sample_depth_default")]
    sample_depth: u8,
    #[serde(default = "n_fft_default")]
    n_fft: u16,
    #[serde(default = "n_filt_default")]
    n_filt: u8,
    #[serde(default = "n_mfcc_default")]
    n_mfcc: u8,
    #[serde(default = "use_delta_default")]
    use_delta: bool,
    #[serde(default = "threshold_config_default")]
    threshold_config: Vec<(f32, f32)>,
    #[serde(default = "threshold_center_default")]
    threshold_center: f32,
    //#[serde(default = "vectorizer_default")]
    //vectorizer: Vectorizer
}
fn buffer_t_default() -> f32 {1.5}
fn window_t_default() -> f32 {0.1}
fn hop_t_default() -> f32 {0.05}
fn sample_rate_default() -> u16 {16000}
fn sample_depth_default() -> u8 {2}
fn n_fft_default() -> u16 {512}
fn n_filt_default() -> u8 {20}
fn n_mfcc_default() -> u8 {13}
fn use_delta_default() -> bool {false}
fn threshold_config_default() -> Vec<(f32, f32)> {vec![(6.0,4.0)]}
fn threshold_center_default() -> f32 {0.2}
//fn vectorizer_default() -> {Vectorizer.mfccs}

impl PreciseParams {
    fn buffer_samples(&self) -> u32 {
        let samples = self.to_samples(self.buffer_t);
        self.hop_samples() * (samples / self.hop_samples())
    }
    
    fn n_features(&self) -> u32 {
        1 + ((
                ((self.buffer_samples() - self.window_samples()) as f32) / (self.hop_samples() as f32)
        ).floor() as u32)
    }

    fn hop_samples(&self) -> u32 {
        self.to_samples(self.hop_t)
    }

    fn window_samples(&self) -> u32 {
        self.to_samples(self.window_t)
    }
    
    fn to_samples(&self, t: f32) -> u32 {
        ((self.sample_rate as f32) * t + 0.5) as u32
    }
}

pub struct Precise {
    model: PreciseModel,
    mfccs: Array2<f32>,
    params: PreciseParams,
    decoder: ThresholdDecoder,
    window_audio: Vec<i16>
}


impl Precise {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, PreciseError> {
        let params = Self::load_params(model_path.as_ref())?;

        let model = PreciseModel::new(model_path)?;
        let decoder = ThresholdDecoder::new(params.threshold_config.clone(), params.threshold_center,200, -4, 4);
        let mfccs = Array2::zeros((params.n_features() as usize, params.n_mfcc as usize));
        Ok(Self{model, mfccs, params, decoder, window_audio: Vec::new()})
    }

    fn load_params(model: &Path) -> Result<PreciseParams, PreciseError> {
        let file = File::open(model.with_extension("tflite.params")).map_err(PreciseError::ParamsLoadError)?;
        from_reader(BufReader::new(file)).map_err(PreciseError::ParamsJsonError)
    }

    fn vectorize_raw(&self, raw: Vec<i16>) -> Array2<f32> {
        const CHUNK_MS: u32 = 10;
        const OUT_MEL_SAMPLES: usize = 20;
        
        let samples = ((self.params.sample_rate as u32 * CHUNK_MS ) / 1000) as usize;
        let chunks =raw.chunks(samples);
        let mut trans = Transform::new(
            self.params.sample_rate as usize,
            samples
        ).nfilters(OUT_MEL_SAMPLES, 40)
        .normlength(10);

        let mels = chunks.map::<Vec<_>,_>(|c|{
            let mut out = vec![0.0; OUT_MEL_SAMPLES * 3];
            if c.len() == samples {
                trans.transform(c, &mut out);
            }
            else { // Needed for the last one
                let mut s = c.to_vec();
                s.resize(samples, 0);
                trans.transform(&s, &mut out);
            }
            out.into_iter().map(|f|f as f32).collect()
        }).flatten().collect::<Vec<f32>>();
        let num_sets = (mels.len() as f32/(OUT_MEL_SAMPLES) as f32).ceil() as usize;

        Array2::from_shape_vec((num_sets, OUT_MEL_SAMPLES), mels).unwrap()
        
    }

    fn update_vectors(&mut self, stream: &[i16]) -> Array2<f32> {
        self.window_audio.extend(stream.iter().cloned());
        if self.window_audio.len() >= self.params.window_samples() as usize {
            let mut new_features = self.vectorize_raw(self.window_audio.clone());
            self.window_audio = self.window_audio[new_features.len() * self.params.hop_samples() as usize..].to_vec(); // Remove old samples
            if new_features.len() > self.mfccs.dim().0 {
                new_features = new_features.slice(s![new_features.len() - self.mfccs.dim().0..,..]).to_owned();
            }

            self.mfccs = concatenate![Axis(0), self.mfccs.slice(s![new_features.len()..,..]).to_owned(), new_features];
        }

        self.mfccs.clone()
    }


    pub fn update(&mut self, audio: &[i16]) -> Result<f32, PreciseError> {
        // Do this first so that we are not borrowed mutable twice
        let mfccs = self.update_vectors(audio);
        let out = self.decoder.decode(self.model.predict(&mfccs)?);
        Ok(out)
    }

    pub fn clear(&mut self) {
        self.window_audio.clear();
        self.mfccs = Array2::zeros((self.params.n_features() as usize, self.params.n_mfcc as usize));
    }
}


#[derive(Debug, Clone)]
pub(crate) struct ThresholdDecoder {
    min_out: i32,
    out_range: i32,
    cd: Array1<f32>,
    center: f32
}

impl ThresholdDecoder {
    fn new(mu_stds: Vec<(f32,f32)>, center: f32, resolution: u32, min_z: i8, max_z: i8 ) -> Self {
        const ERR_NAN: &str = "Tried to compare a NaN";
        const ERR_EMPTY: &str = "Mu Stds is empty";

        let min_out =  mu_stds.iter().map(|(mu,std)|mu + min_z as f32 * std).min_by(
            |a,b| a.partial_cmp(b).expect(ERR_NAN)
        ).expect(ERR_EMPTY) as i32;
        let max_out =  mu_stds.iter().map(|(mu,std)|mu + max_z as f32 * std).max_by(
            |a,b| a.partial_cmp(b).expect(ERR_NAN).reverse()
        ).expect(ERR_EMPTY) as i32;


        let mut cd = Self::calc_pd(mu_stds,
            resolution as f32,
            min_out as f32,
            max_out as f32,
            (max_out - min_out) as usize
        );

        cd.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

        Self{
            min_out,
            out_range: max_out - min_out,
            cd,
            center
        }
    }

    pub(crate) fn pdf(x: &Array1<f32>, mu: f32, std: f32) -> Array1<f32> {
        if std == 0.0 {
            x * 0.0
        }
        else {
            let a1 = 1.0 / (std * (2.0 * PI).sqrt());
            
            let b1 =  -(x - mu).mapv(|v|v.powf(2.0));
            let b2 =  b1 / (2.0 * std.powf(2.0));
            a1 * (b2.mapv(|v|v.exp()))
        }
    }

    fn asigmoid(x: f32) -> f32 {
        -(1.0 / (x - 1.0)).log(E)
    }

    fn  calc_pd(mu_stds: Vec<(f32, f32)>, resolution: f32, min_out: f32, max_out: f32, out_range: usize) -> Array1<f32> {
        let points = Array1::linspace(min_out, max_out, resolution as usize * out_range);
        let len_mu_stds = mu_stds.len() as f32; // Save this early, we are moving the data later

        let pdf = mu_stds.into_iter()
            .map(|(mu,std)|Self::pdf(&points, mu, std)).collect::<Vec<_>>();
        let res = Array1::zeros(pdf[0].dim());
        let pdf_views = pdf.iter().map(|v|v.view()).collect::<Vec<_>>();
        let pdf_2 = stack(Axis(0),&pdf_views).unwrap();
            //.fold(res,|res,x| res + x);
        let pdf_old = pdf.iter().fold(res,|res:Array1<f32> ,x| res + x);

        let pdf_res = pdf_2.sum_axis(Axis(0));
        println!("are equal {}, len_2: {}, len_old: {}", pdf_res == pdf_old, pdf_res.len(), pdf_old.len());

        pdf_2.sum_axis(Axis(0))/(resolution * len_mu_stds)
    }

    pub(crate) fn decode(&self, raw_output: f32) -> f32 {
        if raw_output == 1.0 || raw_output == 0.0 {
            raw_output
        }
        else {
            let cp = if self.out_range == 0 {
                (raw_output > self.min_out as f32) as u8 as f32
            }
            else {
                let ratio = (Self::asigmoid(raw_output) - self.min_out as f32) / self.out_range as f32;
                let ratio = ratio.max(0.0).min(1.0);
                let len_cd = self.cd.len() as f32; // Change if cd size changes
                self.cd[( (ratio * (len_cd - 1.0)) + 0.5) as usize]
            };

            if cp < self.center {
                0.5 * cp / self.center
            }
            else {
                0.5 + 0.5 * (cp - self.center)/(1.0 - self.center)
            }

        }
    }
}

struct PreciseModel {
    model: FlatBufferModel,
}

impl PreciseModel {
    fn new<P: AsRef<Path>>(model: P) -> Result<Self, PreciseError> {
        Ok(Self {model: FlatBufferModel::build_from_file(model.as_ref())?})
    }

    fn predict(&mut self, mfccs: &Array2<f32>) -> Result<f32, PreciseError> {
        const ERR_MFCC: &str = "MFCC data is not contiguous or not in standard order";

        let resolver = BuiltinOpResolver::default();
        let builder = InterpreterBuilder::new(&self.model, &resolver)?;
        let mut interpreter = builder.build()?;

        interpreter.allocate_tensors()?;

        let inputs = interpreter.inputs().to_vec();
        let input_index = inputs[0];

        let outputs = interpreter.outputs().to_vec();
        let output_index = outputs[0];

        interpreter.tensor_data_mut(input_index)?[0..mfccs.len()].copy_from_slice(mfccs.as_slice().expect(ERR_MFCC));

        interpreter.invoke()?;

        let raw_out: &[f32] = interpreter.tensor_data(output_index)?;
        Ok(raw_out[0])
    }
}

#[cfg(test)]
mod tests {
    use super::{Precise, ThresholdDecoder};
    use ndarray::{array, aview1};

    fn load_samples() -> Vec<i16> {
        let mut reader = hound::WavReader::open("test.wav").unwrap();
        let samples: Vec<i16> = reader.samples().map(|e|e.unwrap()).collect();
        samples
    }
    #[test]
    fn test_positive() {
        let mut precise = Precise::new("hey_mycroft.tflite").unwrap();
        println!("{:?}", precise.update(&load_samples()).unwrap());
    }
    #[test]
    fn test_pdf() {
        assert_eq!(ThresholdDecoder::pdf(&array![0.0f32], 0.0, 0.0), aview1(&[0.0]));
        assert_eq!(ThresholdDecoder::pdf(&array![0.0f32, 1.0, 2.0, 3.0, 5.0],2.5,1.4), aview1(&[0.057855975, 0.16051139, 0.2673528, 0.2673528, 0.057855975]));
    }
}
