from torch import optim

from datasets.coco import CocoDetection
from torchvision import transforms
# from transforms import presets
# from optimizer import param_dict

# Commonly changed training configurations
num_epochs = 12   # train epochs
batch_size = 2    # total_batch_size = #GPU x batch_size
num_workers = 4   # workers for pytorch DataLoader
pin_memory = True # whether pin_memory for pytorch DataLoader
print_freq = 50   # frequency to print logs
starting_epoch = 0
max_norm = 0.1    # clip gradient norm
device = 'cuda'
clip_grad = 0.02
clip_mode = 'agc'

output_dir = None  # path to save checkpoints, default for None: checkpoints/{model_name}
find_unused_parameters = False  # useful for debugging distributed training

# define dataset for train
coco_path = "data/coco"  # /PATH/TO/YOUR/COCODIR
# train_transform = presets.detr  # see transforms/presets to choose a transform
data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(0.5)]),
    "val": transforms.Compose([transforms.ToTensor()])
}

train_dataset = CocoDetection("/mnt/d/CoCoStuff2017", dataset="train", transforms=data_transform['train'])
test_dataset = CocoDetection("/mnt/d/CoCoStuff2017", dataset="val", transforms=data_transform['val'])

# model config to train
model_path = "configs/salience_detr_mobilenetv4_medium_800_1333.py"

# specify a checkpoint folder to resume, or a pretrained ".pth" to finetune, for example:
# checkpoints/salience_detr_resnet50_800_1333/train/2024-03-22-09_38_50
# checkpoints/salience_detr_resnet50_800_1333/train/2024-03-22-09_38_50/best_ap.pth
resume_from_checkpoint = None

learning_rate = 1e-4  # initial learning rate
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[10], gamma=0.1)

# This define parameter groups with different learning rate
# param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)