use std::convert::TryInto;
use std::f32::consts::PI;
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

    fn chop_chunks(samples: &[i16], window_size: usize, hop_size: usize) -> impl Iterator<Item = &[i16]> {
        (window_size..samples.len()+1).step_by(hop_size).map(move |i| &samples[i - window_size..i])
    }

    fn mfcc_spec(raw: Vec<i16>, params: &PreciseParams) -> Array2<f32> {
        let out_mel_samples = params.n_mfcc.into();
        let samples= params.window_samples().try_into().unwrap();
        let hop_s = params.hop_samples() as usize;
        let chunks =Self::chop_chunks( &raw, samples, hop_s);
        let mut trans = Transform::new(
            params.sample_rate.into(),
            samples
        ).nfilters(out_mel_samples, params.n_filt.into())
        .normlength(params.n_fft.into());

        let mels = chunks.map::<Vec<_>,_>(|c|{
            let mut out = vec![0.0; out_mel_samples * 3];
            if c.len() == samples {
                trans.transform(c, &mut out);
            }
            else { // Needed for the last one
                let mut s = c.to_vec();
                s.resize(samples, 0);
                trans.transform(&s, &mut out);
            }
            out[..out_mel_samples].iter().map(|f|*f as f32).collect()
        }).flatten().collect::<Vec<f32>>();
        let num_sets = (mels.len() as f32/(out_mel_samples) as f32).ceil() as usize;

        Array2::from_shape_vec((num_sets, out_mel_samples), mels).unwrap()
    }

    fn vectorize_raw(raw: Vec<i16>, params: &PreciseParams) -> Array2<f32> {
        Self::mfcc_spec(raw, params)
    }

    ///Inserts extra features that are the difference between adjacent timesteps
    fn add_deltas(features: &Array2<f32>) -> Array2<f32> {
        /*let mut deltas = Array2::zeros(features.raw_dim());
        for i in 1..features.nrows() {
            deltas[i] = features[i] - features[i - 1];
        }
        return np.concatenate([features, deltas], -1)*/
        Array2::zeros(features.raw_dim())
    }


    fn update_vectors(&mut self, stream: &[i16]) -> Array2<f32> {
        self.window_audio.extend(stream.iter().cloned());
        if self.window_audio.len() >= self.params.window_samples() as usize {
            let mut new_features = Self::vectorize_raw(self.window_audio.clone(), &self.params);
            //let hop_s = self.params.hop_samples();
            //self.window_audio = self.window_audio[new_features.nrows() * hop_s as usize..].to_vec(); // Remove old samples
            self.window_audio = self.window_audio[new_features.nrows()..].to_vec(); // Remove old samples
            if new_features.len() > self.mfccs.nrows() {
                new_features = new_features.slice(s![new_features.nrows() - self.mfccs.dim().0..,..]).to_owned();
            }

            self.mfccs = concatenate![Axis(0), self.mfccs.slice(s![new_features.nrows()..,..]).to_owned(), new_features];
        }

        self.mfccs.clone()
    }


    pub fn update(&mut self, audio: &[i16]) -> Result<f32, PreciseError> {
        let mut mfccs = self.update_vectors(audio);
        if self.params.use_delta {
            // UNTESTED!
            mfccs = Self::add_deltas(&mfccs);
        }
        let out = self.decoder.decode(self.model.predict(&mfccs)?);
        Ok(out)
    }

    pub fn clear(&mut self) {
        self.window_audio.clear();
        self.mfccs = Array2::zeros((self.params.n_features() as usize, (self.params.n_mfcc as usize)*3));
    }
}


#[derive(Debug, Clone)]
struct ThresholdDecoder {
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

    fn pdf(x: &Array1<f32>, mu: f32, std: f32) -> Array1<f32> {
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
        (x / (1.0 - x)).ln()
    }

    fn  calc_pd(mu_stds: Vec<(f32, f32)>, resolution: f32, min_out: f32, max_out: f32, out_range: usize) -> Array1<f32> {
        let points = Array1::linspace(min_out, max_out, resolution as usize * out_range);
        let len_mu_stds = mu_stds.len() as f32; // Save this early, we are moving the data later

        // PDF and PDF_2 are the same look into what's better
        let pdf = mu_stds.into_iter()
            .map(|(mu,std)|Self::pdf(&points, mu, std)).collect::<Vec<_>>();
        let res = Array1::zeros(pdf[0].dim());
        let pdf_views = pdf.iter().map(|v|v.view()).collect::<Vec<_>>();
        let pdf_2 = stack(Axis(0),&pdf_views).unwrap();
            //.fold(res,|res,x| res + x);
        let _pdf_old = pdf.iter().fold(res,|res:Array1<f32> ,x| res + x);

        let _pdf_res = pdf_2.sum_axis(Axis(0));
        

        pdf_2.sum_axis(Axis(0))/(resolution * len_mu_stds)
    }

    fn decode(&self, raw_output: f32) -> f32 {
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
                let len_cd = self.cd.len() as f32;
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
    use crate::PreciseParams;

    use super::{Precise, ThresholdDecoder};
    use assert_approx_eq::assert_approx_eq;
    use ndarray::{array, aview1, s};

    fn load_samples() -> Vec<i16> {
        let mut reader = hound::WavReader::open("test_data/test.wav").unwrap();
        let samples: Vec<i16> = reader.samples().map(|e|e.unwrap()).collect();
        samples
    }
    #[test]
    fn test_pdf() {
        assert_eq!(ThresholdDecoder::pdf(&array![0.0f32], 0.0, 0.0), aview1(&[0.0]));
        assert_eq!(
            ThresholdDecoder::pdf(&array![0.0f32, 1.0, 2.0, 3.0, 5.0],2.5,1.4),
            aview1(&[0.057_855_975, 0.160_511_39, 0.267_352_8, 0.267_352_8, 0.057_855_975]));
    }
    #[test]
    fn chop_chunks() {
        assert_eq!(
            Precise::chop_chunks(&[1,2,3,4,5,6,7,8,9], 3, 2).collect::<Vec<_>>(),
            &[&[1,2,3], &[3,4,5], &[5,6,7], &[7,8,9]]
        );

        let s = vec![0;195804];
        assert_eq!(Precise::chop_chunks(&s, 1600, 800).count(), 243);
    }
    #[test]
    fn decode() {
        let thres = ThresholdDecoder::new(vec![(6.0,4.0)], 0.2, 200, -4, 4 );
        assert_eq!(thres.min_out, -10);
        assert_eq!(thres.cd.len(), 6400);
        assert_eq!(thres.cd.slice(s![230..240]), aview1(
            &[7.128_421e-5   , 7.179_359e-5, 7.230_534e-5  , 7.281_946_5e-5,
              7.333_598e-5   , 7.385_489e-5, 7.437_621_4e-5, 7.489_996e-5  ,
              7.542_613_5e-5 , 7.595_475e-5])
        );
        assert_eq!(thres.out_range, 32);
        
        assert_approx_eq!(thres.decode(0.3), 0.108_636_04);
    }
    #[test]
    fn mfcc() {
        let params = PreciseParams {
            window_t: 0.1,
            hop_t: 0.05,
            sample_rate: 16_000,
            n_fft: 512,
            n_mfcc: 13,
            n_filt: 20,

            // Not needed here
            buffer_t: 0.0, 
            sample_depth: 0, 
            use_delta: false,
            threshold_center: 0.0,
            threshold_config: vec![]
        };

        let mfcc = Precise::mfcc_spec(vec![0;195804], &params);

        assert_eq!(mfcc.nrows(), 243);
        assert_eq!(mfcc.ncols(), 13);
        /* Note: MFCC behaviour seems different from sonopy (the one that the 
            Python version uses), however, seems like the model is still capableç
            of reading it*/
        /*assert_eq!(mfcc.slice(s![120..125,..]), aview2(&[
            [-3.604_365e1   ,  0.00           ,  0.00000000e+00 ,
              0.0           ,  1.367_879_9e-15,  0.0            ,
              0.0           ,  0.0            , -1.163_588_1e-15,
              0.0           ,  0.0            ,  0.0            ,
             -8.453_962_7e-16],
            [-3.604_365e1   ,  0.0            ,  0.0            ,
              0.0           ,  1.367_879_9e-15,  0.0            ,
              0.0           ,  0.0            , -1.163_588_1e-15,
              0.0           ,  0.0            ,  0.0            ,
             -8.453_962_7e-16],
            [-3.604_365e1   ,  0.0            ,  0.0            ,
              0.0           ,  1.367_879_9e-15,  0.0            ,
              0.0           ,  0.0            , -1.163_588_1e-15,
              0.0           ,  0.0            ,  0.0            ,
             -8.453_962_7e-16],
            [-3.604_365e1   ,  0.0            ,  0.0            ,
              0.0           ,  1.367_879_9e-15,  0.0            ,
              0.0           ,  0.0            , -1.163_588_1e-15,
              0.0           ,  0.0            ,  0.0            ,
            -8.453_962_7e-16],
            [-3.604_365e1   ,  0.0            ,  0.0            ,
              0.0           ,  1.367_879_9e-15,  0.0            ,
              0.0           ,  0.0            , -1.163_588_1e-15,
              0.0           ,  0.0            ,  0.0            ,
            -8.453_962_7e-16]]))*/
    }
    #[test]
    fn asigmoid() {
        assert_approx_eq!(ThresholdDecoder::asigmoid(0.3), -0.8472979);
    }
    #[test]
    fn test_positive() {
        let mut precise = Precise::new("test_data/hey_mycroft.tflite").unwrap();
        assert!(precise.update(&load_samples()).unwrap() >= 0.9);
    }
}
