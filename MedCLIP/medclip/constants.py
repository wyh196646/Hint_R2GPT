'''drawn from Gloria github: https://github.com/marshuang80/gloria
'''

BERT_TYPE = 'emilyalsentzer/Bio_ClinicalBERT'
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
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}

COVID_TASKS = [
    'Normal',
    'COVID',
]
COVID_CLASS_PROMPTS = {
    'COVID': {
        'adjective': ['patchy','confluent'],
        'description': ['ground glass'],
        'subtype': ['opacity', 'consolidation'],
        'location': ['in peripheral', 'in mid', 'in lower'],
    }
}

RSNA_TASKS = [
    'Normal',
    'Pneumonia',
]
RSNA_CLASS_PROMPTS = {
    'Pneumonia': {
        'adjective': ['round', 'early', 'focal', 'multifocal', 'small', ''],
        'subtype': ['bacterial', 'viral', 'mycoplasma', ''],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left middle lobe",
            "at the right middle lobe",
            ""
        ]
    }
}

WEIGHTS_NAME = 'pytorch_model.bin'

# store the URL of pretrained weights, `dev` needs to change to `main` after merging it to main branch.
PRETRAINED_URL_MEDCLIP_RESNET = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_resnet_weight.txt'
PRETRAINED_URL_MEDCLIP_VIT = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_vit_weight.txt'




MIMIC_CLASS_PROMPTS = {
    "No Finding": {
        "severity": [""],
        "subtype": [
            "no acute cardiopulmonary process",
            "lungs are clear",
            "no focal consolidation, effusion, or pneumothorax",
            "cardiomediastinal silhouette within normal limits",
            "no acute osseous abnormality",
        ],
        "location": [""],
    },

    "Enlarged Cardiomediastinum": {
        "severity": ["", "mild", "moderate", "severe", "borderline"],
        "subtype": [
            "enlarged cardiomediastinal silhouette",
            "widened mediastinum",
            "prominent mediastinal contours",
            "upper limits of normal cardiomediastinal silhouette",
        ],
        "location": [""],
    },

    "Cardiomegaly": {
        "severity": ["", "mild", "moderate", "severe", "borderline"],
        "subtype": [
            "cardiac silhouette mildly enlarged",
            "heart size is enlarged",
            "stable cardiomegaly",
            "prominent cardiac silhouette",
        ],
        "location": [""],
    },

    "Lung Lesion": {
        "severity": ["", "small", "subtle", "large"],
        "subtype": [
            "solitary pulmonary nodule",
            "pulmonary mass",
            "rounded opacity",
            "cavitary lesion",
            "spiculated nodule",
        ],
        "location": [
            "at the right upper lobe",
            "at the left upper lobe",
            "at the right lower lobe",
            "at the left lower lobe",
            "perihilar region",
            "at the lung bases",
        ],
    },

    "Lung Opacity": {
        "severity": ["", "increased", "decreased", "diffuse"],
        "subtype": [
            "airspace opacity",
            "patchy opacity",
            "interstitial opacities",
            "reticulonodular opacities",
            "hazy parenchymal opacities",
        ],
        "location": [
            "at the bilateral lungs",
            "at the perihilar regions",
            "at the lower lung zones",
            "at the upper lung zones",
            "at the peripheral lungs",
        ],
    },

    "Edema": {
        "severity": ["", "trace", "mild", "moderate", "severe", "improving", "worsening"],
        "subtype": [
            "pulmonary interstitial edema",
            "pulmonary edema",
            "perihilar edema",
            "Kerley B lines",
        ],
        "location": [""],
    },

    "Consolidation": {
        "severity": ["", "increased", "improved", "new", "resolving"],
        "subtype": [
            "airspace consolidation",
            "patchy consolidation",
            "retrocardiac consolidation",
            "lobar consolidation",
        ],
        "location": [
            "at the left lower lobe",
            "at the right lower lobe",
            "at the right middle lobe",
            "at the upper lung zones",
            "at the lung bases",
        ],
    },

    "Pneumonia": {
        "severity": ["", "suspected", "likely", "improving"],
        "subtype": [
            "pneumonic consolidation",
            "airspace disease compatible with pneumonia",
            "infectious process",
            "aspiration pattern",
        ],
        "location": [
            "at the right lower lobe",
            "at the left lower lobe",
            "at the lingula",
            "at the right middle lobe",
            "multifocal",
        ],
    },

    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
        ],
        "location": [
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the upper lung zones",
        ],
    },

    "Pneumothorax": {
        "severity": ["", "small", "moderate", "large", "trace"],
        "subtype": [
            "apical pneumothorax",
            "loculated pneumothorax",
            "tension physiology suspected",
            "pleural line with absent lung markings",
        ],
        "location": [
            "at the right apex",
            "at the left apex",
            "at the lateral chest",
            "at the anterior chest",
            "bilateral",
        ],
    },

    "Pleural Effusion": {
        "severity": ["", "tiny", "small", "moderate", "large", "decreased", "increased", "stable"],
        "subtype": [
            "bilateral pleural effusions",
            "subpulmonic effusion",
            "loculated effusion",
            "layering pleural fluid",
        ],
        "location": ["left", "right", "bilateral", "at the costophrenic angles"],
    },

    "Pleural Other": {
        "severity": ["", "mild", "marked"],
        "subtype": [
            "pleural thickening",
            "pleural plaques",
            "pleural calcifications",
            "post-pleurodesis changes",
            "empyema (pleural collection)",
            "apical pleural capping",
        ],
        "location": ["left", "right", "bilateral"],
    },

    "Fracture": {
        "severity": ["", "acute", "subacute", "chronic", "healed", "minimally displaced", "displaced"],
        "subtype": [
            "rib fracture",
            "clavicle fracture",
            "scapular fracture",
            "sternal fracture",
            "vertebral compression deformity",
        ],
        "location": [
            "posterior ribs",
            "lateral ribs",
            "anterior ribs",
            "right clavicle",
            "left clavicle",
            "thoracic spine",
        ],
    },

    "Support Devices": {
        "severity": [""],
        "subtype": [
            "endotracheal tube",
            "enteric tube",
            "central venous catheter",
            "chest tube",
            "pacemaker leads",
            "ICD",
            "sternal wires",
            "valve prosthesis",
            "ECMO cannula",
            "intra-aortic balloon pump",
        ],
        "location": [
            "tip projects over the carina",
            "tip in the stomach",
            "tip over the SVC",
            "tip at the cavoatrial junction",
            "left subclavian approach",
            "right IJ approach",
        ],
    },
}
