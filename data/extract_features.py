import torch 
# from utils import compute_logits
import clip
import logging
import os
from tqdm import tqdm
from data.transforms.default import build_transform


log = logging.getLogger(__name__)



def makedirs(path, verbose=False):
    '''Make directories if not exist.'''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if verbose:
            print(path + " already exists.")


def compute_features(dataloader, forward_method):
    img_path_list = []
    features_list = []
    idxs_list = []

    for img, aug_1, idxs, label, img_path in tqdm(dataloader):
        with torch.no_grad():
            features = forward_method(img)
        
        img_path_list.extend(list(img_path))
        features_list.append(features)
        idxs_list.append(idxs)

    features_list = torch.cat(features_list, dim=0)
    idxs_list = torch.cat(idxs_list, dim=0).long()
    return img_path_list, idxs_list, features_list


def get_image_encoder_dir(feature_dir, clip_encoder):
    image_encoder_path = os.path.join(
        feature_dir,
        'image',
        get_image_encoder_name(clip_encoder)
    )
    return image_encoder_path

def get_image_encoder_name(clip_encoder):
    return clip_encoder.replace("/", "-")

def get_view_name(image_augmentation, image_views=1):
    name = f"aug_{image_augmentation}"
    if image_augmentation != "none":
        assert image_views > 0
        name += f"_view_{image_views}"
    return name

def get_image_features_path(dataset,
                            is_train,
                            feature_dir,
                            clip_encoder,
                            image_augmentation,
                            image_views=1):
    image_features_path = os.path.join(
        get_image_encoder_dir(feature_dir, clip_encoder),
        os.path.join(dataset, "train" if is_train else "test"),
        get_view_name(image_augmentation, image_views),
        f"features.pth")
    return image_features_path


def prepare_image_features(clip_model, dataset, 
                           image_augmentation='none', image_views=1):
    #define forward:
    def img_forward(img):
        image_features = clip_model.encode_image(img.to(clip_model.ln_final.weight.device))
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )
        return image_features
    
    #create a new dataloader:
    assert hasattr(dataset, 'transform'), "Dataset must have a transform attribute"
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=8,
            drop_last=False,
    )
    # get image features save path
    image_features_path = get_image_features_path(
        dataset.dataset_name,
        is_train=dataset.train,
        feature_dir="Features/",
        clip_encoder=clip_model.encoder_name,
        image_augmentation=image_augmentation,
        image_views=image_views
    )

    makedirs(os.path.dirname(image_features_path))
    
    # Check if (image) features are saved already
    if os.path.exists(image_features_path):
        log.info(f"Features already saved at {image_features_path}, load it")
        image_features = torch.load(image_features_path)
    else:
        log.info(f"Saving features to {image_features_path}")

        train_transform = build_transform(image_augmentation)
        test_transform = build_transform('none')
        dataset.transform = train_transform if dataset.train else test_transform

        if image_augmentation == 'none':
            num_views = 1
        else:
            num_views = image_views
        assert num_views > 0, "Number of views must be greater than 0"

        img_paths, idxs, features = compute_features(loader, img_forward)
        dataset.transform = None
        image_features = {'img_paths': img_paths, 'img_features': features.cpu(), 'idxs': idxs.cpu()}
        torch.save(image_features, image_features_path)


    return image_features
