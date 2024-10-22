import logging

import datasets
from datasets.config import HF_DATASETS_CACHE

logger = logging.getLogger(__name__)


class Dataset:
    NAME: str  # unique name for saving
    HF_PATH: str  # Huggingface URL
    TOKENIZER: str  # tokenizer used during preprocessing
    BPE: str  # subword encoding used during preprocessing

    REGISTRY = {}

    @staticmethod
    def register(*dataset_names):
        def wrapper(cls):
            for dataset in dataset_names:
                if dataset in cls.REGISTRY:
                    raise ValueError(f"Dataset {dataset} has been registered.")
                cls.REGISTRY[dataset] = cls
            return cls

        return wrapper

    @classmethod
    def load(cls, task, **kwargs):
        datasets.utils.logging.set_verbosity_info()
        datasets.utils.logging.disable_propagation()  # to prevent double logging

        logger.info(f"Loading dataset {cls.NAME}: {kwargs}")
        destination = cls._load(task, **kwargs)
        logger.info("Dataset Loaded!")
        return destination

    @classmethod
    def _load(cls, task, **kwargs):
        raise NotImplementedError(f"Please implement the `_load` method in the {cls.__name__} class.")

    @classmethod
    def _get_extracted_dir(cls, *args, **kwargs):
        dataset_builder = datasets.load_dataset_builder(
            cls.HF_PATH,
            *args,
            **kwargs,
            token=True,
            cache_dir=HF_DATASETS_CACHE,
        )

        download_config = datasets.DownloadConfig(
            cache_dir=dataset_builder._cache_downloaded_dir,
            use_etag=False,
            token=dataset_builder.token,
            storage_options=dataset_builder.storage_options,
        )
        dl_manager = datasets.DownloadManager(
            dataset_name=dataset_builder.dataset_name,
            download_config=download_config,
            data_dir=dataset_builder.config.data_dir,
            base_path=dataset_builder.base_path,
            record_checksums=False,
        )
        data_dir = dl_manager.download_and_extract(dataset_builder.config.data_url)

        return data_dir
