use std::path::Path;
use tract_tensorflow::prelude::*;
use tract_core::TractError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PreciseError {
    #[error("Failure while operating model")]
    ModelError(#[from] TractError),

    #[error("With shape")]
    ShapeErrror(#[from]ndarray::ShapeError)
}

pub struct Precise {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl Precise {
    pub fn new(model: &Path) -> Result<Self, PreciseError> {
        let tf = tensorflow();
        let model = tf.model_for_path(model)?;
        let model = model
        .with_input_names(&["import/net_input"])?
        .with_output_names(&["import/net_output"])?
        .into_optimized()?
        .into_runnable()?;
        Ok(Self{model})
    }

    pub fn update(&self, audio: &[i16]) -> Result<Tensor, PreciseError> {
        let input = tract_ndarray::arr1(audio).into_shape((10,100))?.into_tensor();
        let result = self.model.run(tvec![input])?[0].nth(0)?;
        let a = result.to_array_view::<i16>()?;
        println!("test: {:?}", a);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use super::Precise;
    #[test]
    fn it_works() {
        let path = Path::new("test.wav");
        let precise = Precise::new(Path::new("hey-microft.pb")).unwrap();
        let mut reader = hound::WavReader::open(path).unwrap();
        let samples: Vec<i16> = reader.samples().map(|e|e.unwrap()).collect();
        println!("{:?}", precise.update(&samples).unwrap());
    }
}
