# Tract-precise

A reimplementation of the [precise](https://github.com/MycroftAI/mycroft-precise)
by [Mycroft](https://mycroft.ai) hotword listener (just the decoder, though) 
using [Rust](https://www.rust-lang.org/) and [Tract](https://github.com/sonos/tract).

# Usage

Add this to your `Cargo.toml`

```toml
[dependencies]
precise-rs = {git = "https://github.com/sheosi/precise-rs"}
```

# How it works

A little example:

```rust
fn load_samples() -> Vec<i16> {
    let mut reader = hound::WavReader::open("test.wav").unwrap();
    let samples: Vec<i16> = reader.samples().map(|e|e.unwrap()).collect();
    samples
}

fn main() {
    const WAKEWORD_THRESHOLD : f32 = 0.8;

    let mut precise = Precise::new("hey-mycroft.pb").unwrap();

    if precise.update(&load_samples()).unwrap() > 0.8 {
        println!("Wakeword recognized");
    }
}
```

This example loads the `test.wav` Wav file and runs that audio through the `hey-mycroft.pb` tflite model, note that also `hey-mycroft.pb.args` needs to be present, which is the configuration for that model. You can find more info [here](https://github.com/sheosi/precise-rs/wiki/Arguments).

# Obtaining the code

Before cloning you need to have git LFS (Large Files Storage).

First install git LFS, for Debian/Ubuntu it is:

```shell
sudo apt install git-lfs
```
For fedora: 
```shell
sudo dnf install git-lfs
```

Then, no matter which OS you are under you need to initialize git LFS:

```shell
git lfs install
```

# What works what not

## Vectorizer

Only "mfcc" vectorizer is available and is one that's based very much like speechpy's, so anything using sonopy's mfcc vectorizer should work. This means that any model not requiring any vectorizer or setting it as "2" will work (that's speechpy's), any model using "1" should work (more testing is needed), and any model using "0" (mel_spec) won't work for now.


## Delta

Delta is not implemented right now.

# License 
Licensed under Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0).

# Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, shall be dual licensed as using the Apache-2.0 license, without any additional terms or conditions.
