use std::path::Path;

use memmap2::{Mmap, MmapMut};

use crate::entry::entry_point::{OperationError, OperationResult};
use crate::madvise;
use crate::types::ScalarQuantizationConfig;
use crate::vector_storage::quantized::quantized_vectors_base::QuantizedVectors;
use crate::vector_storage::quantized::scalar_quantized::{
    ScalarQuantizedVectors, ScalarQuantizedVectorsConfig,
};

pub struct QuantizedMmapStorage {
    mmap: Mmap,
}

pub struct QuantizedMmapStorageBuilder {
    mmap: MmapMut,
    cursor_pos: usize,
}

impl quantization::EncodedStorage for QuantizedMmapStorage {
    fn get_vector_data(&self, index: usize, vector_size: usize) -> &[u8] {
        &self.mmap[vector_size * index..vector_size * (index + 1)]
    }

    fn from_file(
        path: &Path,
        quantized_vector_size: usize,
        vectors_count: usize,
    ) -> std::io::Result<QuantizedMmapStorage> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        madvise::madvise(&mmap, madvise::get_global())?;

        let expected_size = quantized_vector_size * vectors_count;
        if mmap.len() == expected_size {
            Ok(Self { mmap })
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "Loaded storage size {} is not equal to expected size {expected_size}",
                    mmap.len()
                ),
            ))
        }
    }

    fn save_to_file(&self, _path: &Path) -> std::io::Result<()> {
        // do nothing because mmap is already saved
        Ok(())
    }
}

impl quantization::EncodedStorageBuilder<QuantizedMmapStorage> for QuantizedMmapStorageBuilder {
    fn build(self) -> QuantizedMmapStorage {
        self.mmap.flush().unwrap();
        let mmap = self.mmap.make_read_only().unwrap(); // TODO: remove unwrap
        QuantizedMmapStorage { mmap }
    }

    fn push_vector_data(&mut self, other: &[u8]) {
        self.mmap[self.cursor_pos..self.cursor_pos + other.len()].copy_from_slice(other);
        self.cursor_pos += other.len();
    }
}

impl QuantizedMmapStorageBuilder {
    pub fn new(
        path: &Path,
        vectors_count: usize,
        quantized_vector_size: usize,
    ) -> std::io::Result<Self> {
        let encoded_storage_size = quantized_vector_size * vectors_count;
        path.parent().map(std::fs::create_dir_all);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        file.set_len(encoded_storage_size as u64)?;
        let mmap = unsafe { MmapMut::map_mut(&file) }?;
        madvise::madvise(&mmap, madvise::get_global())?;
        Ok(Self {
            mmap,
            cursor_pos: 0,
        })
    }
}

pub fn create_scalar_quantized_vectors_mmap<'a>(
    vectors: impl IntoIterator<Item = &'a [f32]> + Clone,
    config: &ScalarQuantizationConfig,
    vector_parameters: &quantization::VectorParameters,
    data_path: &Path,
) -> OperationResult<Box<dyn QuantizedVectors>> {
    let quantized_vector_size =
        quantization::EncodedVectorsU8::<QuantizedMmapStorage>::get_quantized_vector_size(
            vector_parameters,
        );
    let storage_builder = QuantizedMmapStorageBuilder::new(
        data_path,
        vector_parameters.count,
        quantized_vector_size,
    )?;
    let quantized_vectors = quantization::EncodedVectorsU8::encode(
        vectors,
        storage_builder,
        vector_parameters,
        config.quantile,
    )
    .map_err(|e| OperationError::service_error(format!("Cannot quantize vector data: {e}")))?;

    let quantized_vectors_config = ScalarQuantizedVectorsConfig {
        quantization_config: config.clone(),
        vector_parameters: vector_parameters.clone(),
    };

    Ok(Box::new(ScalarQuantizedVectors::new(
        quantized_vectors,
        quantized_vectors_config,
    )))
}