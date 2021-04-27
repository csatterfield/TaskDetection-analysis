import pandas as pd
participants = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18", "P19"]
for participant in participants:
    print(participant)
    with open(f"../archives/{participant}/appdata.csv") as f:
        lines = f.readlines()
    
    split_lines = [x.split(",") for x in lines]
    fixed_split_lines = []
    for line in split_lines:
        cur = []
        for i in range(3):
            cur.append(line[i])
        word = ""
        for i in range(3, len(line)):
            word = word + line[i]
        cur.append(word.strip())
        fixed_split_lines.append(cur)
    
    cols = fixed_split_lines[0]
    fixed_split_lines = fixed_split_lines[1:]

    df = pd.DataFrame(fixed_split_lines, columns=cols)
    df = df[df["AppName"] != "Collector"] 
    df = df[df["WindowTitle"] != "Untitled"]
    df = df[df["WindowTitle"] != ""]
    df["StartTime"] = pd.to_datetime(df["StartTime"], unit='s') - pd.Timedelta(7, unit='h')
    df["EndTime"] = pd.to_datetime(df["EndTime"], unit='s') - pd.Timedelta(7, unit='h')
    df.to_csv(f"../archives/{participant}/appdata_fixed.csv", index=False)