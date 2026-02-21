const BACKGROUND_COLOR: u8 = 255;

/// Classification of pixels in an image used for edge thinning.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum Edge {
    /// The pixel does not contain the foreground color.
    Empty = 0,
    /// The pixel contains the foreground color.
    Filled = 1,
    /// The pixel is not a valid location within the image.
    DoesNotExist,
}

use image::Luma;

/// Struct with information describing the surrounding pixels.
pub struct NeighborInfo {
    /// The number of neighbor pixels with non-background color.
    pub filled: u8,
    /// The number of surrounding pixels.
    pub neighbors: u8,
    /// An array containing the [status](crate::Edge) of the neighboring pixels.
    pub edge_status: [Edge; 8],
}

/// Calculate and return a [`NeighborInfo`](crate::neighbors::NeighborInfo)
/// struct which contains the number of occupied neighbor pixels, number of
/// neighbor pixels, and the [edge status](crate::Edge) of the neighboring
/// pixels.
pub fn get_neighbor_info(
    img: &image::GrayImage,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
) -> NeighborInfo {
    let mut filled = 0;
    let mut neighbors = 0;

    let p9 = if y > 0 && x > 0 {
        neighbors += 1;
        if img.get_pixel(x - 1, y - 1)[0] != BACKGROUND_COLOR {
            filled += 1;
            Edge::Filled
        } else {
            Edge::Empty
        }
    } else {
        Edge::DoesNotExist
    };
    let p2 = if y > 0 {
        neighbors += 1;
        if img.get_pixel(x, y - 1)[0] != BACKGROUND_COLOR {
            filled += 1;
            Edge::Filled
        } else {
            Edge::Empty
        }
    } else {
        Edge::DoesNotExist
    };
    let p3 = if y > 0 && x < u32::MAX && x + 1 < width {
        neighbors += 1;
        if img.get_pixel(x + 1, y - 1)[0] != BACKGROUND_COLOR {
            filled += 1;
            Edge::Filled
        } else {
            Edge::Empty
        }
    } else {
        Edge::DoesNotExist
    };
    let p8 = if x > 0 {
        neighbors += 1;
        if img.get_pixel(x - 1, y)[0] != BACKGROUND_COLOR {
            filled += 1;
            Edge::Filled
        } else {
            Edge::Empty
        }
    } else {
        Edge::DoesNotExist
    };
    let p4 = if x < u32::MAX && x + 1 < width {
        neighbors += 1;
        if img.get_pixel(x + 1, y)[0] != BACKGROUND_COLOR {
            filled += 1;
            Edge::Filled
        } else {
            Edge::Empty
        }
    } else {
        Edge::DoesNotExist
    };
    let p7 = if x > 0 && y < u32::MAX && y + 1 < height {
        neighbors += 1;
        if img.get_pixel(x - 1, y + 1)[0] != BACKGROUND_COLOR {
            filled += 1;
            Edge::Filled
        } else {
            Edge::Empty
        }
    } else {
        Edge::DoesNotExist
    };
    let p6 = if y < u32::MAX && y + 1 < height {
        neighbors += 1;
        if img.get_pixel(x, y + 1)[0] != BACKGROUND_COLOR {
            filled += 1;
            Edge::Filled
        } else {
            Edge::Empty
        }
    } else {
        Edge::DoesNotExist
    };
    let p5 = if x < u32::MAX && x + 1 < width && y < u32::MAX && y + 1 < height {
        neighbors += 1;
        if img.get_pixel(x + 1, y + 1)[0] != BACKGROUND_COLOR {
            filled += 1;
            Edge::Filled
        } else {
            Edge::Empty
        }
    } else {
        Edge::DoesNotExist
    };

    NeighborInfo {
        filled,
        neighbors,
        edge_status: [p2, p3, p4, p5, p6, p7, p8, p9],
    }
}

pub fn thin_image_edges(img_in: &image::GrayImage) -> image::GrayImage {
    let mut img = img_in.clone();
    let mut pixels_to_remove = Vec::new();
    let mut phase_one = true;

    loop {
        // Mark pixels to remove
        for (x, y, p) in img.enumerate_pixels() {
            if p.0[0] == BACKGROUND_COLOR {
                continue;
            }

            let info = get_neighbor_info(&img, img.width(), img.height(), x, y);
            let [p2, p3, p4, p5, p6, p7, p8, p9] = info.edge_status;

            if !(2..=7).contains(&info.filled) || info.neighbors != 8 {
                continue;
            }

            let mut transitions = 0;
            for pair in [p2, p3, p4, p5, p6, p7, p8, p9, p2].windows(2) {
                if pair[0] == Edge::Empty && pair[1] == Edge::Filled {
                    transitions += 1;
                }
            }

            if !(transitions == 1 || transitions == 2) {
                continue;
            }

            if phase_one {
                if transitions == 1 {
                    if (p2 == Edge::Empty || p4 == Edge::Empty || p6 == Edge::Empty)
                        && (p4 == Edge::Empty || p6 == Edge::Empty || p8 == Edge::Empty)
                    {
                        pixels_to_remove.push((x, y));
                    }
                } else {
                    if ((p2 == Edge::Filled && p4 == Edge::Filled)
                        && (p6 == Edge::Empty && p7 == Edge::Empty && p8 == Edge::Empty))
                        || ((p4 == Edge::Filled && p6 == Edge::Filled)
                            && (p2 == Edge::Empty && p8 == Edge::Empty && p9 == Edge::Empty))
                    {
                        pixels_to_remove.push((x, y));
                    }
                }
            } else {
                if transitions == 1 {
                    if (p2 == Edge::Empty || p4 == Edge::Empty || p8 == Edge::Empty)
                        && (p2 == Edge::Empty || p6 == Edge::Empty || p8 == Edge::Empty)
                    {
                        pixels_to_remove.push((x, y));
                    }
                } else {
                    if ((p2 == Edge::Filled && p8 == Edge::Filled)
                        && (p4 == Edge::Empty && p5 == Edge::Empty && p6 == Edge::Empty))
                        || ((p6 == Edge::Filled && p8 == Edge::Filled)
                            && (p2 == Edge::Empty && p3 == Edge::Empty && p4 == Edge::Empty))
                    {
                        pixels_to_remove.push((x, y));
                    }
                }
            }
        }

        phase_one = !phase_one;

        // Replace marked pixels with background color to thin the edges
        for &(x, y) in &pixels_to_remove {
            img.put_pixel(x, y, Luma([BACKGROUND_COLOR]));
        }

        if pixels_to_remove.is_empty() {
            return img;
        }

        pixels_to_remove.clear();
    }
}
