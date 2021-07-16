use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use cgmath::InnerSpace;
use std::{convert::TryInto, fs::File, io::BufWriter, path::Path};

#[rustfmt::skip]
const AINV: [[f32; 16]; 16] = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,], 
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,], 
    [-3.0, 3.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,], 
    [2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,], 
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,], 
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,], 
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 3.0, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0,], 
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,], 
    [-3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,], 
    [0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, -1.0, 0.0,], 
    [9.0, -9.0, -9.0, 9.0, 6.0, 3.0, -6.0, -3.0, 6.0, -6.0, 3.0, -3.0, 4.0, 2.0, 2.0, 1.0,],
    [-6.0, 6.0, 6.0, -6.0, -3.0, -3.0, 3.0, 3.0, -4.0, 4.0, -2.0, 2.0, -2.0, -2.0, -1.0, -1.0,],
    [2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,], 
    [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,],
    [-6.0, 6.0, 6.0, -6.0, -4.0, -2.0, 4.0, 2.0, -3.0, 3.0, -3.0, 3.0, -2.0, -1.0, -2.0, -1.0,],
    [4.0, -4.0, -4.0, 4.0, 2.0, 2.0, -2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 1.0, 1.0, 1.0, 1.0,],
];

// Compute coeffs using inverse A matrix
fn compute_coeff(x: &[f32; 16]) -> [f32; 16] {
    let mut alpha: [f32; 16] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    for i in 0..16 {
        for j in 0..16 {
            alpha[i] += AINV[i][j] * x[j];
        }
    }

    alpha
}

/// Bitmap representation
#[derive(Clone)]
struct Bitmap<T> {
    pub size: (usize, usize),
    pub data: Vec<T>,
}
impl<T> Bitmap<T> {
    pub fn pixel(&self, p: (usize, usize)) -> &T {
        assert!(p.0 < self.size.0);
        assert!(p.1 < self.size.1);

        &self.data[p.0 * self.size.1 + p.1]
    }
    pub fn pixel_warp(&self, p: (usize, usize)) -> &T {
        let x = p.0.rem_euclid(self.size.0);
        let y = p.1.rem_euclid(self.size.1);
        &self.data[x * self.size.1 + y]
    }
    pub fn pixel_mut(&mut self, p: (usize, usize)) -> &mut T {
        assert!(p.0 < self.size.0);
        assert!(p.1 < self.size.1);
        &mut self.data[p.0 * self.size.1 + p.1]
    }
}
type Bitmap2 = Bitmap<cgmath::Vector2<f32>>;
type Bitmap3 = Bitmap<cgmath::Vector3<f32>>;
impl Bitmap3 {
    pub fn normalize(&mut self) {
        for v in &mut self.data {
            let l = v.magnitude();
            *v = *v / l;
        }
    }

    pub fn read_exr(filename: &str) -> Self {
        // Open the EXR file.
        let mut file = std::fs::File::open(filename).unwrap();
        let mut input_file = openexr::InputFile::new(&mut file).unwrap();
        println!("Channel informations: ");
        for c in input_file.header().channels() {
            let c = c.unwrap();
            println!("{}: {:?}", c.0, c.1);
        }

        // Get the image dimensions, so we know how large of a buffer to make.
        let (width, height) = input_file.header().data_dimensions();
        let size = (width as usize, height as usize);

        let pixel_type = input_file
            .header()
            .channels()
            .nth(0)
            .unwrap()
            .unwrap()
            .1
            .pixel_type;

        // Depending of the precision, load the frame buffer and convert back to f32
        let data = match pixel_type {
            openexr::PixelType::UINT => panic!("Uint is unsupported"),
            openexr::PixelType::FLOAT => {
                let mut data = vec![(0.0f32, 0.0f32, 0.0f32); (width * height) as usize];
                {
                    let mut fb = openexr::FrameBufferMut::new(width, height);
                    fb.insert_channels(&[("R", 0.0), ("G", 0.0), ("B", 0.0)], &mut data);

                    // Read pixel data from the file.
                    input_file.read_pixels(&mut fb).unwrap();
                }
                // Conversion
                data.into_iter()
                    .map(|v| cgmath::Vector3::new(v.0, v.1, v.2))
                    .collect()
            }
            openexr::PixelType::HALF => {
                let mut data = vec![
                    (
                        half::f16::from_f32(0.0f32),
                        half::f16::from_f32(0.0f32),
                        half::f16::from_f32(0.0f32)
                    );
                    (width * height) as usize
                ];
                {
                    let mut fb = openexr::FrameBufferMut::new(width, height);
                    fb.insert_channels(&[("R", 0.0), ("G", 0.0), ("B", 0.0)], &mut data);

                    // Read pixel data from the file.
                    input_file.read_pixels(&mut fb).unwrap();
                }
                // Conversion
                data.into_iter()
                    .map(|v| cgmath::Vector3::new(v.0.to_f32(), v.1.to_f32(), v.2.to_f32()))
                    .collect()
            }
        };

        Bitmap { size, data }
    }

    // Read LDR image ()
    pub fn read_ldr_image(filename: &str) -> Self {
        // The image that we will render
        let image_ldr = image::open(filename)
            .unwrap_or_else(|_| panic!("Impossible to read image: {}", filename));
        let image_ldr = image_ldr.to_rgb8();
        let size = (image_ldr.width() as usize, image_ldr.height() as usize);
        let mut data = vec![cgmath::Vector3::new(0.0, 0.0, 0.0); (size.0 * size.1) as usize];
        for x in 0..size.0 {
            for y in 0..size.1 {
                let p = image_ldr.get_pixel(x as u32, y as u32);
                data[(y * size.0 + x) as usize] = cgmath::Vector3::new(
                    (f32::from(p[0]) / 255.0) * 2.0 - 1.0,
                    (f32::from(p[1]) / 255.0) * 2.0 - 1.0,
                    (f32::from(p[2]) / 255.0) * 2.0 - 1.0,
                );
            }
        }

        Bitmap { size, data }
    }

    // Read file (LDF or OpenEXR)
    pub fn read(filename: &str) -> Self {
        let ext = match std::path::Path::new(filename).extension() {
            None => panic!("No file extension provided"),
            Some(x) => std::ffi::OsStr::to_str(x).expect("Issue to unpack the file"),
        };
        match ext {
            "exr" => Bitmap::read_exr(filename),
            _ => {
                // Try the default implementation support
                Bitmap::read_ldr_image(filename)
            }
        }
    }
}

// Specialization for bitmap containing coeff files
// These functions are mostly to debug or output intermediate states
impl Bitmap<[f32; 16]> {
    // Compute the ratio between two coeff bitmaps
    #[allow(dead_code)]
    pub fn div(&self, other: &Self) -> Self {
        let mut res = Self {
            size: self.size,
            data: vec![[0.0; 16]; self.size.0 * self.size.1],
        };

        for x in 0..self.size.0 {
            for y in 0..self.size.1 {
                let p = res.pixel_mut((x, y));
                let o1 = self.pixel((x, y));
                let o2 = other.pixel((x, y));
                for i in 0..16 {
                    p[i] = o1[i] / o2[i];
                }
            }
        }

        res
    }

    // Load coeff bitmap (Linqi's format)
    #[allow(dead_code)]
    pub fn load(filename: &str, size: (usize, usize)) -> (Self, Self) {
        let f = File::open(Path::new(filename)).unwrap();
        let mut f = std::io::BufReader::new(f);

        let mut res0 = Self {
            size,
            data: vec![[0.0; 16]; size.0 * size.1],
        };
        let mut res1 = Self {
            size,
            data: vec![[0.0; 16]; size.0 * size.1],
        };
        for j in 0..2 {
            for i in 0..16 {
                // Do (x * size + y)
                for x in 0..size.0 {
                    for y in 0..size.1 {
                        let p = if j == 0 {
                            res0.pixel_mut((x, y))
                        } else {
                            res1.pixel_mut((x, y))
                        }; // Do (y * size + x)
                        let v = f.read_f32::<LittleEndian>().unwrap();
                        p[i] = v;
                    }
                }
            }
        }

        (res0, res1)
    }

    // Save the coeff in a multi-channel EXR
    // So it is easy to inspect inside the EXR Viewer (like tev)
    #[allow(dead_code)]
    pub fn save_exr(&self, filename: &str) {
        // Create a file to write to.  The `Header` determines the properties of the
        // file, like resolution and what channels it has.
        let mut file = std::fs::File::create(filename).unwrap();
        let mut output_file = openexr::ScanlineOutputFile::new(
            &mut file,
            openexr::Header::new()
                .set_resolution(self.size.0 as u32, self.size.1 as u32)
                .add_channel("C0", openexr::PixelType::FLOAT)
                .add_channel("C1", openexr::PixelType::FLOAT)
                .add_channel("C2", openexr::PixelType::FLOAT)
                .add_channel("C3", openexr::PixelType::FLOAT)
                .add_channel("C4", openexr::PixelType::FLOAT)
                .add_channel("C5", openexr::PixelType::FLOAT)
                .add_channel("C6", openexr::PixelType::FLOAT)
                .add_channel("C7", openexr::PixelType::FLOAT)
                .add_channel("C8", openexr::PixelType::FLOAT)
                .add_channel("C9", openexr::PixelType::FLOAT)
                .add_channel("C10", openexr::PixelType::FLOAT)
                .add_channel("C11", openexr::PixelType::FLOAT)
                .add_channel("C12", openexr::PixelType::FLOAT)
                .add_channel("C13", openexr::PixelType::FLOAT)
                .add_channel("C14", openexr::PixelType::FLOAT)
                .add_channel("C15", openexr::PixelType::FLOAT),
        )
        .unwrap();

        // Remap data
        let data = (0..16)
            .map(|i| self.data.iter().map(|c| c[i]).collect::<Vec<f32>>())
            .collect::<Vec<_>>();
        {
            // Insert data and save
            let mut fb = openexr::FrameBuffer::new(self.size.0 as u32, self.size.1 as u32);
            for i in 0..16 {
                fb.insert_channels(&[&format!("C{}", i)], &data[i]);
            }
            output_file.write_pixels(&fb).unwrap();
        }
    }

    // Helper function to save a given channel into a file
    pub fn save_channel(&self, file: &mut std::io::BufWriter<File>, c: usize) {
        for d in &self.data {
            file.write_f32::<LittleEndian>(d[c]).unwrap();
        }
    }

    // Compute the bicubic interpolation using the coefficients
    pub fn get_normal(&self, uv: (f32, f32)) -> f32 {
        let x = uv.0 * self.size.0 as f32;
        let y = uv.1 * self.size.1 as f32;
        let x_int = (x.floor() as usize).rem_euclid(self.size.0);
        let y_int = (y.floor() as usize).rem_euclid(self.size.1);

        let mut v = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                let c = j * 4 + i;
                let coeff = self.pixel((x_int, y_int))[c];
                v += coeff * (x - x.floor()).powi(i as i32) * (y - y.floor()).powi(j as i32);
            }
        }
        v
    }
}

fn main() {
    // TODO: Add clap for command line uses
    // Read image
    // -- Flakes
    let filename = "data/flakes.exr";
    // let filename_coeff = "data/flakes.exr.coeff";
    // -- Scratchs
    // let filename = "data/scratch_wave_0.05.exr";
    // let filename_coeff = "data/scratch_wave_0.05.exr.coeff";

    // Load the normal map image
    let mut image = Bitmap3::read(filename);
    {
        dbg!(image
            .data
            .iter()
            .map(|v| v.x)
            .max_by(|x, y| x.partial_cmp(&y).unwrap()));
        dbg!(image
            .data
            .iter()
            .map(|v| v.x)
            .min_by(|x, y| x.partial_cmp(&y).unwrap()));
    }
    image.normalize(); // Normalize the normals

    // Compute coeffs
    println!("Compute coeffs ... ");
    let mut c0 = Bitmap::<[f32; 16]> {
        size: image.size,
        data: vec![[0.0; 16]; image.size.0 * image.size.1],
    };
    let mut c1 = Bitmap::<[f32; 16]> {
        size: image.size,
        data: vec![[0.0; 16]; image.size.0 * image.size.1],
    };
    {
        // Helpers
        let f = |x: usize, y: usize| *image.pixel_warp((x, y));
        let f_x = |x: usize, y: usize| {
            (image.pixel_warp((x + 1, y)) - image.pixel_warp((x - 1, y))) * 0.5
        };
        let f_y = |x: usize, y: usize| {
            (image.pixel_warp((x, y + 1)) - image.pixel_warp((x, y - 1))) * 0.5
        };
        let f_xy = |x: usize, y: usize| {
            let v0 = f_x(x, y - 1);
            let v1 = f_x(x, y + 1);
            (v1 - v0) * 0.5
        };

        for i in 0..image.size.0 {
            for j in 0..image.size.1 {
                let x1 = i;
                let y1 = j;
                let x2 = x1 + 1;
                let y2 = y1 + 1;

                let v = [
                    f(x1, y1),
                    f(x2, y1),
                    f(x1, y2),
                    f(x2, y2),
                    f_x(x1, y1),
                    f_x(x2, y1),
                    f_x(x1, y2),
                    f_x(x2, y2),
                    f_y(x1, y1),
                    f_y(x2, y1),
                    f_y(x1, y2),
                    f_y(x2, y2),
                    f_xy(x1, y1),
                    f_xy(x2, y1),
                    f_xy(x1, y2),
                    f_xy(x2, y2),
                ];
                *c0.pixel_mut((x1, y1)) = compute_coeff(
                    &v.iter()
                        .map(|v| v.x) //(((v.x * 255.0) as i32) as f32 / 255.0))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                );
                *c1.pixel_mut((x1, y1)) = compute_coeff(
                    &v.iter()
                        .map(|v| v.y)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                );
            }
        }

        ///////////////////////
        // Debug code:
        // This can load original coeff file and save it back to multi channel EXR.
        // Also compute the ratio between the computed coeffs and the original one
        // to spot any differences.
        ////////////////////////

        // println!("Write coeffs (exr)... ");
        // c0.save_exr("c0.exr");
        // c1.save_exr("c1.exr");
        // {
        //     let (c0ref, c1ref) = Bitmap::<[f32; 16]>::load(filename_coeff, image.size);
        //     c0ref.save_exr("c0ref.exr");
        //     c1ref.save_exr("c1ref.exr");
        //     let c0div = c0ref.div(&c0);
        //     c0div.save_exr("c0div.exr");
        // }

        // Save coeff in the binary form
        println!("Write coeffs ... ");
        let file = File::create(Path::new("test.coeff")).unwrap();
        let mut file = BufWriter::new(file);
        for c in 0..16 {
            c0.save_channel(&mut file, c);
        }
        for c in 0..16 {
            c1.save_channel(&mut file, c);
        }
    }

    println!("Compute bound files...");
    // Compute bounds: For now for 32 triangles (4*4 tiles)
    const MAX_SEG: u32 = 1 << 2;
    {
        let mut max = Bitmap2 {
            size: image.size,
            data: vec![cgmath::Vector2::new(0.0, 0.0); image.size.0 * image.size.1],
        };
        let mut min = Bitmap2 {
            size: image.size,
            data: vec![cgmath::Vector2::new(0.0, 0.0); image.size.0 * image.size.1],
        };

        for i in 0..image.size.0 {
            for j in 0..image.size.1 {
                for k in 0..MAX_SEG {
                    for l in 0..MAX_SEG {
                        let x1 = ((k as f32) / MAX_SEG as f32 + i as f32) * image.size.0 as f32;
                        let y1 = ((l as f32) / MAX_SEG as f32 + j as f32) * image.size.1 as f32;
                        let x2 =
                            ((k as f32 + 1.0) / MAX_SEG as f32 + i as f32) * image.size.0 as f32;
                        let y2 =
                            ((l as f32 + 1.0) / MAX_SEG as f32 + j as f32) * image.size.1 as f32;

                        // Compute min/max for the bounds
                        for (c, coeffs) in [&c0, &c1].iter().enumerate() {
                            let v0 = coeffs.get_normal((x1, y1));
                            let v1 = coeffs.get_normal((x2, y1));
                            let v2 = coeffs.get_normal((x1, y2));
                            let v3 = coeffs.get_normal((x2, y2));
                            let v_max = max.pixel((i, j))[c];
                            let v_min = min.pixel((i, j))[c];

                            max.pixel_mut((i, j))[c] = v_max.max(v0).max(v1).max(v2).max(v3);
                            min.pixel_mut((i, j))[c] = v_min.min(v0).min(v1).min(v2).min(v3);
                        }
                    }
                }
            }
        }

        println!("Write out bounds...");
        let file = File::create(Path::new("test.bounds")).unwrap();
        let mut file = BufWriter::new(file);
        for v in &max.data {
            file.write_f32::<LittleEndian>(v.x).unwrap();
        }
        for v in &min.data {
            file.write_f32::<LittleEndian>(v.x).unwrap();
        }
        for v in &max.data {
            file.write_f32::<LittleEndian>(v.y).unwrap();
        }
        for v in &min.data {
            file.write_f32::<LittleEndian>(v.y).unwrap();
        }
    }

    println!("DONE!");
}
