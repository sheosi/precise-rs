use std::f32::consts::{E, PI};
use std::fs::File;
use std::io::{Read, BufReader};
use std::path::Path;

use serde_json::from_reader;
use serde::{Deserialize, Serialize};
use mfcc::Transform;
use ndarray::{Array, Array2, ArrayBase, Dim, OwnedRepr, s};
use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs, Tensor};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PreciseError {
    #[error("Couldn't open model file")]
    FileError(#[from]std::io::Error),
    #[error("Failure while operating model")]
    TensorflowError(#[from]tensorflow::Status),
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

#[derive(Debug, Clone)]
pub struct ThresholdDecoder {
    min_out: i32,
    out_range: i32,
    cd: (f32, f32),
    center: f32
}

impl ThresholdDecoder {
    fn new(mu_stds: Vec<(f32,f32)>, center: f32, resolution: u32, min_z: i8, max_z: i8 ) -> Self {
        println!("mu_stds: {:?}", &mu_stds);
        let min_out =  mu_stds.iter().map(|(mu,std)|mu + min_z as f32 * std).min_by(
            |a,b| a.partial_cmp(b).expect("Tried to compare a NaN")
        ).unwrap() as i32;
        let max_out =  mu_stds.iter().map(|(mu,std)|mu + max_z as f32 * std).max_by(
            |a,b| a.partial_cmp(b).expect("Tried to compare a NaN").reverse()
        ).unwrap() as i32;


        let (a,b) = Self::calc_pd(mu_stds,
            resolution as f32,
            min_out as f32,
            max_out as f32,
            (max_out - min_out) as usize
        );
        let cd = (a, a+b);

        Self{
            min_out,
            out_range: max_out - min_out,
            cd,
            center
        }
    }

    fn pdf(x: ArrayBase<OwnedRepr<f32>, Dim<[usize;1]>>, mu: f32, std: f32) -> ArrayBase<OwnedRepr<f32>, Dim<[usize;1]>> {
        if std == 0.0 {
            x *0.0
        }
        else {
            (1.0 / (std * (2.0 * PI).sqrt())) * (-(x - mu).mapv(|v|v.powf(2.0 / (2.0 * std.powf(2.0)))).mapv(|v|v.exp()))
        }
    }

    fn asigmoid(x: f32) -> f32 {
        -(1.0 / (x - 1.0).log(E))
    }

    fn  calc_pd(mu_stds: Vec<(f32, f32)>, resolution: f32, min_out: f32, max_out: f32, out_range: usize) -> (f32, f32) {
        println!("min_out: {}, max_out: {}, out_range: {}", min_out, max_out, out_range);
        let points = Array::linspace(min_out, max_out, resolution as usize * out_range);
        let ms_len = mu_stds.len()  as f32;
        let pdf = mu_stds.into_iter().map(|(mu,std)|Self::pdf(points.clone(), mu, std)).fold((0.0,0.0),|(a1, a2),x| (a1 + x[0], a2 + x[1]));
        let apply = |v|v as f32/(resolution * ms_len);
        (apply(pdf.0), apply(pdf.1))
    }

    pub fn decode(&self, raw_output: f32) -> f32 {
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
                let len_cd = 2.0; // Change if cd size changes
                if (ratio * (len_cd - 1.0) + 0.5) == 0.0{
                    self.cd.0
                }
                else {
                    self.cd.1
                }
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

#[derive(Debug)]
pub struct Precise {
    graph: Graph,
    mfccs: Array2<f32>,
    params: PreciseParams,
    decoder: ThresholdDecoder,
    window_audio: Vec<i16>
}

impl Precise {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, PreciseError> {
        let mut graph = Graph::new();
        let mut proto = Vec::new();
        File::open(&model_path)?.read_to_end(&mut proto)?;
        graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
    
        let params = Self::load_params(model_path.as_ref())?;
        let decoder = ThresholdDecoder::new(params.threshold_config.clone(), params.threshold_center,200, -4, -4);
        let mfccs = Array::zeros((params.n_features() as usize, params.n_mfcc as usize));
        Ok(Self{graph, mfccs, params, decoder, window_audio: Vec::new()})
    }

    fn load_params(model: &Path) -> Result<PreciseParams, PreciseError> {
        let file = File::open(model.with_extension("pb.params")).unwrap();
        Ok(from_reader(BufReader::new(file)).unwrap())
    }

    fn vectorize_raw(&self, raw: Vec<i16>) -> Vec<f32> {
        let mut trans = Transform::new(
            self.params.sample_rate as usize,
            self.params.window_samples() as usize
        );
        let mut out = Vec::new();
        trans.transform(raw.as_slice(), &mut out);
        out.into_iter().map(|f|f as f32).collect()
    }

    fn update_vectors(&mut self, stream: &[i16]) -> Array2<f32> {
        self.window_audio.extend(stream.iter().cloned());
        if self.window_audio.len() >= self.params.window_samples() as usize {
            let mut new_features = self.vectorize_raw(self.window_audio.clone());
            self.window_audio = self.window_audio[new_features.len() * self.params.hop_samples() as usize..].to_vec();
            if new_features.len() > self.mfccs.len() {
                new_features = new_features[new_features.len() - self.mfccs.len()..].to_vec();
            }
            self.mfccs = self.mfccs.slice(s![new_features.len()..,..]).to_owned();//self.mfccs.unwrap()[new_features.len()..].to_vec().extend(new_features.iter().cloned());
        }

        self.mfccs.clone()
    }


    pub fn update(&mut self, audio: &[i16]) -> Result<f32, PreciseError> {
        
        let mfccs = self.update_vectors(audio);
        let input = Tensor::from(mfccs);
        
        let session = Session::new(&SessionOptions::new(), &self.graph)?;

        let mut args = SessionRunArgs::new();
        args.add_feed(&self.graph.operation_by_name_required("net_input")?, 0, &input);
        let out = args.request_fetch(&self.graph.operation_by_name_required("net_output")?, 0);
        session.run(&mut args)?;
        
        let raw_out: f32 = args.fetch(out)?[0];
        
        let out = self.decoder.decode(raw_out);
        println!("raw: {:?}, decoded: {:?}, session: {:?}", raw_out, out, session);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::Precise;

    fn load_samples() -> Vec<i16> {
        let mut reader = hound::WavReader::open("test.wav").unwrap();
        let samples: Vec<i16> = reader.samples().map(|e|e.unwrap()).collect();
        samples
    }
    #[test]
    fn test_positive() {
        let mut precise = Precise::new("hey-mycroft.pb").unwrap();
        println!("{:?}", precise.update(&load_samples()).unwrap());
    }
}
