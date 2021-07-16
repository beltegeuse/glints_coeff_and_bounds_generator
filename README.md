# Description

Rust implementation for computing the coefficient and bound files for the paper titled: "
Rendering Glints on High-Resolution Normal-Mapped Specular Surfaces" Linqi et al., 2014, SIGGRAPH

This code produces files that are slightly different from some coefficient and bound files provided by the authors. Therefore, please revise the implementation before using it to generate results.

# Usage

```shell
glint_coeff_bounds 0.1

USAGE:
    glints_coeff_and_bounds_generator <filename> <output>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

ARGS:
    <filename>    filename for normal-map or coeff file (depending of the mode)
    <output>      output name
```

Example: 
```shell
cargo run --release -- data/flakes.exr flakes
```

# TODO

- [ ] Re-enable OpenEXR export of the coefficient (for inspection)
- [ ] Investigate the differences between coefficients (especially f_xy derivatives)
- [ ] Validate/Test bound file
