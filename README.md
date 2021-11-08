# Precise-rs

A reimplementation of the [precise](https://github.com/MycroftAI/mycroft-precise)
by [Mycroft](https://mycroft.ai) hotword listener (just the decoder, though) 
using [Rust](https://www.rust-lang.org/) and [tflite-rs](https://github.com/boncheolgu/tflite-rs).

**NOTE:** While precise accepts three vectorizer operations *precise-rs* only replicates what the original names as `mfcc` as of now.

# Usage

Add this to your `Cargo.toml`

```toml
[dependencies]
tract-precise = {git = "https://github.com/sheosi/tract-precise"}
```

# Example

```rust
use precise_rs::Precise;

fn main() {
    let hotword_engine = Precise::new("my_precise_model.tflite");
    let audio_input: &[i16] = get_audio_input();
    let confidence = hotword_engine.update(audio_input);
    if (confidence > 0.8) {
        println!("Heard someone!!");
    }
}
```

**NOTE:** Remember to also have `my_precise_model.tflite.params` next to `my_precise_model.tflite`.

# How it works

This is pretty much a reimplementation of the original [code](https://github.com/MycroftAI/mycroft-precise), it loads the model and gets the mfccs

# Obtaining the code

This uses git LFS which means it needs to be cloned with that installed and ready beforehand.

First install git LFS, for Debian/Ubuntu it is:

```shell
sudo apt install git-lfs
```

Then, no matter which OS you are under you need to initialize git LFS:

```shell
git lfs install
```

# License 
Licensed under Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0).

# Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, shall be dual licensed as using the Apache-2.0 license, without any additional terms or conditions.
