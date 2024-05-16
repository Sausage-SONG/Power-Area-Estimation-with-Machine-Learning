from pathlib import Path
from hashlib import md5
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from tqdm import tqdm
from itertools import repeat
from copy import deepcopy
from shutil import rmtree
import os

def read_contest(csv_file):
    with Path(csv_file).open("r") as f:
        lines = f.readlines()
    lines = map(lambda x: x.strip().split("    "), lines)
    lines = list(map(lambda x: [float(num) for num in x], lines))
    return lines

def extract_config(item):
    config = {}
    config["fetch_width"] = str(int(item[0]))
    config["decode_width"] = str(int(item[1]))

    config["dispatch_width"] = str(int(item[2]))
    config["issue_width"] = str(int(item[3]))
    config["number_of_ISUs"] = str(int(item[4]))

    config["int_dispatch_width"] = str(int(item[5]))
    config["int_issue_width"] = str(int(item[6]))
    config["number_of_int_ISUs"] = str(int(item[7]))

    config["fp_dispatch_width"] = str(int(item[8]))
    config["fp_issue_width"] = str(int(item[9]))
    config["number_of_fp_ISUs"] = str(int(item[10]))

    config["prediction_width"] = str(int(item[11]))
    config["IFU/fetch_buffer"] = str(int(item[12]))
    config["IFU/target_queue"] = str(int(item[13]))

    config["ROB_size"] = str(int(item[14]))

    config["phy_Regs_IRF_size"] = str(int(item[15]))
    config["phy_Regs_FRF_size"] = str(int(item[16]))

    config["load_buffer_size"] = str(int(item[17]))
    config["store_buffer_size"] = str(int(item[18]))

    config["icache/icache_config"] = "{},{}".format(int(item[19]), int(item[20]))
    config["icache/itlb_config"] = "{},{}".format(int(item[21]), int(item[22]))

    config["dcache/dcache_config"] = "{},{}".format(int(item[23]), int(item[24]))
    config["dcache/replacement_policy"] = str(int(item[25]))
    config["dcache/mshr_entries"] = str(int(item[26]))
    config["dcache/dtlb_config"] = "{},{}".format(int(item[27]), int(item[28]))
    return config

def apply_config(config, xml_data):
    for key, value in config.items():
        if "/" in key:
            key = key.split("/")
            parent = xml_data.find(".//*[@name='{}']".format(key[0]))
            child = parent.find(".//*[@name='{}']".format(key[1]))
            child.set("value", value)
        else:
            child = xml_data.find(".//*[@name='{}']".format(key))
            child.set("value", value)
    return xml_data

def process_mcpat_report(rpt_file):
    with Path(rpt_file).open("r") as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.strip(), lines))
    start = lines.index("Core:")
    # end = lines.index("L2")
    # lines = lines[start:end]
    # lines = list(filter(lambda x: "inf" not in x and (x.endswith("W") or x.endswith("mm^2")), lines))
    area = float(lines[start + 1].split(" ")[-2])
    dynamic = float(lines[start + 2].split(" ")[-2])
    sleak = float(lines[start + 3].split(" ")[-2])
    gleak = float(lines[start + 4].split(" ")[-2])
    return area, dynamic, sleak, gleak

def mcpat(args):
    line, tpt, iid, res_file = args
    xml_data = deepcopy(tpt)
    # iid = md5(str(line).encode()).hexdigest()
    config = extract_config(line)
    apply_config(config, xml_data)
    xml_data.write("temp/{}.xml".format(iid))
    os.system("../mcpat -infile temp/{}.xml -print_level 5 > temp/{}.rpt".format(iid, iid))
    area, dynamic, sleak, gleak = process_mcpat_report("temp/{}.rpt".format(iid))
    with Path(res_file).open("a") as f:
        f.write("{},{},{},{},{}\n".format(iid, area, dynamic, sleak, gleak))
    Path("temp/{}.xml".format(iid)).unlink()
    Path("temp/{}.rpt".format(iid)).unlink()


if __name__ == "__main__":
    contest_data = read_contest("contest.csv")
    iids = list(range(1, len(contest_data)+1))
    tpt = ET.parse('template_wo_gem5.xml')

    tmp_dir = Path("temp")
    if tmp_dir.exists():
        rmtree(tmp_dir)
    tmp_dir.mkdir()

    res_file = Path("data1.csv")
    if res_file.exists():
        res_file.unlink()
    with res_file.open("w") as f:
        f.write("id,area,dynamic,sleak,gleak\n")

    # contest_data = contest_data[:]
    # iids = iids[:10]

    with Pool(32) as p:
        list(
            tqdm(
                p.imap(mcpat, zip(contest_data, repeat(tpt), iids, repeat(res_file))),
                total=len(contest_data),
            )
        )