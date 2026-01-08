import os
import glob
from tqdm import tqdm


ROBOFLOW_NAMES = [
    'DP.135 - End all restrictions', 'I.423b - Pedestrian crossing', 'P.102 - No entry', 
    'P.103a - No cars', 'P.104 - No motorcycles', 'P.106a - No trucks', 
    'P.106b - Weight limit for trucks', 'P.107a - No buses', 'P.112 - No pedestrians', 
    'P.115 - Weight limit', 'P.117 - Height limit', 'P.123a - No left turn', 
    'P.123b - No right turn', 'P.124a - No U-turn', 'P.124b - No U-turn for cars', 
    'P.124c - No left turn or U-turn', 'P.124d - No right turn or U-turn', 
    'P.125 - No overtaking', 'P.127 - Speed limit', 'P.128 - No honking', 
    'P.130 - No stopping or parking', 'P.131a - No parking', 'P.137 - No left or right turn', 
    'R.301c - Obligatory left turn', 'R.301d - Obligatory right turn', 
    'R.302a - Right turn only', 'R.302b - Left turn only', 'R.303 - Roundabout', 
    'R.407a - One way', 'R.409 - U-turn allowed', 'R.434 - Bus stop', 
    'S.509a - Safe height info', 'W.201 - Dangerous curve', 'W.202 - Zigzag road', 
    'W.203 - Narrow road', 'W.205a - 4-way intersection', 'W.205b - T-intersection', 
    'W.207 - Non-priority intersection', 'W.224 - Pedestrian crossing ahead', 
    'W.225 - Children crossing', 'W.227 - Construction', 'W.245a - Go slow'
]

MASTER_NAMES_DICT = {
    0: 'DP.135 - End all restrictions',
    1: 'I.408 - Parking allowed',
    2: 'I.423b - Pedestrian crossing',
    3: 'P.102 - No entry',
    4: 'P.103a - No cars',
    5: 'P.103b - No left turn for cars',
    6: 'P.103c - No right turn for cars',
    7: 'P.104 - No motorcycles',
    8: 'P.106a - No trucks',
    9: 'P.106b - Weight limit for trucks',
    10: 'P.107a - No buses',
    11: 'P.112 - No pedestrians',
    12: 'P.115 - Weight limit',
    13: 'P.117 - Height limit',
    14: 'P.123a - No left turn',
    15: 'P.123b - No right turn',
    16: 'P.124a - No U-turn',
    17: 'P.124b - No U-turn for cars',
    18: 'P.124c - No left turn or U-turn',
    19: 'P.125 - No overtaking',
    20: 'P.127 - Speed limit',
    21: 'P.128 - No honking',
    22: 'P.130 - No stopping or parking',
    23: 'P.131a - No parking',
    24: 'P.137 - No left or right turn',
    25: 'UNUSED_P.245a',
    26: 'R.301c - Obligatory left turn',
    27: 'R.301d - Obligatory right turn',
    28: 'MERGED_INTO_26',
    29: 'R.302a - Right turn only',
    30: 'R.302b - Left turn only',
    31: 'R.303 - Roundabout',
    32: 'R.407a - One way',
    33: 'R.409 - U-turn allowed',
    34: 'UNUSED_R.425',
    35: 'R.434 - Bus stop',
    36: 'S.509a - Safe height info',
    37: 'W.201 - Dangerous curve',
    38: 'MERGED_INTO_37',
    39: 'W.202 - Zigzag road',
    40: 'MERGED_INTO_39',
    41: 'W.203 - Narrow road',
    42: 'MERGED_INTO_41',
    43: 'W.205a - 4-way intersection',
    44: 'W.205b - T-intersection',
    45: 'MERGED_INTO_44',
    46: 'W.207 - Non-priority intersection',
    47: 'MERGED_INTO_46',
    48: 'MERGED_INTO_46',
    49: 'W.208 - Yield',
    50: 'W.209 - Traffic lights ahead',
    51: 'W.210 - Railway crossing',
    52: 'W.224 - Pedestrian crossing ahead',
    53: 'W.225 - Children crossing',
    54: 'W.227 - Construction',
    55: 'W.245a - Go slow',
    56: 'P.124d - No right turn or U-turn'
}

id_map = {}
master_name_to_id = {v: k for k, v in MASTER_NAMES_DICT.items()}

print("Building ID Mapping...")
for robo_id, name in enumerate(ROBOFLOW_NAMES):
    if name in master_name_to_id:
        master_id = master_name_to_id[name]
        id_map[robo_id] = master_id
        print(f"   {robo_id}: '{name}' -> {master_id}")
    else:
        print(f"CRITICAL WARNING: Could not find '{name}' in Master List!")


def convert_dataset(source_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    
    label_files = glob.glob(os.path.join(source_folder, "*.txt"))
    print(f"\nProcessing {len(label_files)} files in {source_folder}...")

    for lbl_path in tqdm(label_files):
        filename = os.path.basename(lbl_path)
        out_path = os.path.join(output_folder, filename)
        
        new_lines = []
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            try:
                robo_cls_id = int(parts[0])
                
                if robo_cls_id in id_map:
                    correct_id = id_map[robo_cls_id]
                    new_line = f"{correct_id} " + " ".join(parts[1:]) + "\n"
                    new_lines.append(new_line)
                else:
                    print(f"Unknown Class ID {robo_cls_id} in {filename}")
            except ValueError:
                continue
                
        with open(out_path, 'w') as f:
            f.writelines(new_lines)


convert_dataset(
    source_folder=r"C:/Users/Admin/Downloads/VNTS_new/valid/labels", 
    output_folder=r"C:/Users/Admin/Downloads/VNTS_fixed/val/labels"
)

print("\nConversion Complete. Copy the files from 'VNTS_Roboflow_Fixed' to your main dataset.")