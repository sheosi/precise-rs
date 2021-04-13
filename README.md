# Tract-precise

A reimplementation of the [precise](https://github.com/MycroftAI/mycroft-precise)
by [Mycroft](https://mycroft.ai) hotword listener (just the decoder, though) 
using [Rust](https://www.rust-lang.org/) and [Tract](https://github.com/sonos/tract).

# Usage

Add this to your `Cargo.toml`

```toml
[dependencies]
tract-precise = {git = "https://github.com/sheosi/tract-precise"}
```

# How it works

TODO

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
