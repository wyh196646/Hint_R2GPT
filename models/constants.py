'''drawn from Gloria github: https://github.com/marshuang80/gloria
'''

BERT_TYPE = '/data/wyh21/huggingface/models--emilyalsentzer--Bio_ClinicalBERT'
VIT_TYPE = 'microsoft/swin-tiny-patch4-window7-224'

IMG_SIZE = 224
IMG_MEAN = .5862785803043838
IMG_STD = .27950088968644304

CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": [""],
        "subtype": [
            "There is atelectasis",
        ],
        "location": [
           ""
        ],
    },
    #['Atelectasis','Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    #'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
   "Cardiomegaly": {
       "severity": [""],
         "subtype": [
              "There is cardiomegaly",
         ],
            "location": [
                ""
            ],
    },
    "Consolidation": {
        "severity": [""],
        "subtype": [
            "There is consolidation",
        ],
        "location": [
            ""
        ],
    },
    "Edema": {
        "severity": [""],
        "subtype": [
            "There is edema",
        ],
        "location": [
            ""
        ],
    },
    "Enlarged Cardiomediastinum": {
        "severity": [""],
        "subtype": [
            "There is enlarged cardiomediastinum",
        ],
        "location": [
            ""
        ],
    },
    "Fracture": {
        "severity": [""],
        "subtype": [
            "There is fracture",
        ],
        "location": [
            ""
        ],
    },
    "Lung Lesion": {
        "severity": [""],
        "subtype": [
            "There is lung lesion",
        ],
        "location": [
            ""
        ],
    },
    "Lung Opacity": {
        "severity": [""],
        "subtype": [
            "There is lung opacity",
        ],
        "location": [
            ""
        ],
    },
    "No Finding": {
        "severity": [""],
        "subtype": [
            "There is no finding",
        ],
        "location": [
            ""
        ],
    },
    "Pleural Effusion": {
        "severity": [""],
        "subtype": [
            "There is pleural effusion",
        ],
        "location": [
            ""
        ],
    },
    "Pleural Other": {
        "severity": [""],
        "subtype": [
            "There is pleural other",
        ],
        "location": [
            ""
        ],
    },
    "Pneumonia": {
        "severity": [""],
        "subtype": [
            "There is pneumonia",
        ],
        "location": [
            ""
        ],
    },
    "Pneumothorax": {
        "severity": [""],
        "subtype": [
            "There is pneumothorax",
        ],
        "location": [
            ""
        ],
    },
    "Support Devices": {
        "severity": [""],
        "subtype": [
            "There is support devices",
        ],
        "location": [
            ""
        ],
    },
}


WEIGHTS_NAME = 'pytorch_model.bin'

# store the URL of pretrained weights, `dev` needs to change to `main` after merging it to main branch.
PRETRAINED_URL_MEDCLIP_RESNET = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_resnet_weight.txt'
PRETRAINED_URL_MEDCLIP_VIT = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_vit_weight.txt'