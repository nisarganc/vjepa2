import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import dlimp as dl

# UTN dataset loading
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd
import pyarrow.dataset as ds
from PIL import Image
import io

# DROID dataset loading
import tensorflow_datasets as tfds
import dlimp as dl
import rlds

def load_parquet_episode():   
    "Returns UTN episode numpy arrays" 
    
    # Load LeRobot format UTN dataset
    # TODO: change path to one episode file
    DATASET_PATH = '/home/abut37yx/RCSToolBox/extracted_dataset/data/chunk-000'
    PARTITIONING = ds.partitioning(
                    schema=pa.schema([pa.field("uuid", pa.binary(36))]), 
                    flavor="filename")

    EPISODE = ds.dataset(DATASET_PATH)
    print(EPISODE.schema)

    df = pd.read_parquet(
            DATASET_PATH,
            engine="pyarrow",
            
            # partitioning=PARTITIONING,
            # columns=["uuid", "reward", 
            #           "observation.frames.wrist.rgb", 
            #           "step"],
            # columns={"uuid": "uuid", 
                        # "reward": "reward", 
                        # "observation.frames.wrist.rgb": "observation.frames.wrist.rgb", 
                        # "step": "step"},
            columns={
                "image": pc.field("image"),
                "state": pc.field("state"),
                "actions": pc.field("actions"),
                "frame_index": pc.field("frame_index"),
                "index": pc.field("index"),
                "task_index": pc.field("task_index"),
            },
            dtype_backend="pyarrow",
        )
    # print df keys
    print(df.columns.values)
    
    # load "df["image"]" into numpy array
    np_clips = np.array([np.array(
                        Image.open(io.BytesIO(df["image"][i]['bytes'])
                        )) for i in range(len(df))]) # (N, 256, 256, 3)

    np_states = np.array([np.array(
                        df["state"][i]) for i in range(len(df))]) # (N, 7)

    np_actions = np.array([np.array(
                        df["actions"][i]) for i in range(len(df))]) # (N, 7)
    
    return np_clips, np_states, np_actions

def load_droid_episode():
    "Returns droid episode numpy arrays"

    builder = tfds.builder_from_directory("/mnt/dataset_drive/OpenX_Embodiment/droid_100/1.0.0")
    dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False, num_parallel_reads=1)
    
    # print(builder.info.features) 

    np_states = np.array([])
    np_actions = np.array([])

    for episode in dataset.as_numpy_iterator():

        im = ([np.array(Image.open(io.BytesIO(episode["observation"]["exterior_image_2_left"][i])
                        )) for i in range(len(episode["observation"]["exterior_image_2_left"]))])
        np_clips = np.array(im)

        break
    
    del dataset, builder

    return np_clips, np_states, np_actions