import argparse
import json
from qwikidata.json_dump import WikidataJsonDump
import os
import sys
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

SUPPORT_MODE = ["unchanged", "changed"]


def construct_generation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="unchanged", required=True, choices=SUPPORT_MODE)
    parser.add_argument('--old', type=str, default='20211101')
    parser.add_argument('--new', type=str, default='20211201')
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--combine', type=int, default=0)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10000)

    arg = parser.parse_args()
    return arg


def extraction(month, idx):
    dump_location = f"fact_triplets_raw/wikidata-{month}.json.gz"

    wjd = WikidataJsonDump(dump_location)

    big_list = {}

    for ii, entity_dict in enumerate(wjd):
        small_list = []
        entity_id = entity_dict["title"]
        s = int(entity_id[1:])
        if s < idx*1000000:
            continue
        if s >= idx*1000000 + 1000000:
            break
        texts = entity_dict["section_texts"][0]
        if len(texts) < 2000:
            print(entity_dict)
            print('')
        index = -1
        while True:
            index = texts.find("wikibase-entity", index + 1)
            target = texts[index-200:index]
            idx1 = target.find("\"property\"")
            idx2 = target.find("hash")
            idx3 = target.find("\"id\":\"")
            idx4 = target.find(",\"type\":")
            relation = target[idx1+12: idx2-3]
            objective = target[idx3+6: idx4-2]
            if relation == "" or objective == "":
                pass
            elif "\"},\"type\":" in objective:
                small_list.append([relation, objective[:-10]])
            else:
                small_list.append([relation, objective])
            if index == -1:
                break
        small_list = small_list[:-1]
        semi_result = []
        for i in small_list:
            if i not in semi_result:
                    semi_result.append(i)
        big_list[entity_id] = semi_result

    idx = str(idx)
    output_dir = f"fact_triplets_raw/{month}/{month}_{idx}.json"

    with open(output_dir, "w") as write_json_file:
        json.dump(big_list, write_json_file, indent=4)


def id(old, new, idx, mode):
    old_address = f"fact_triplets_raw/{old}/{old}_{idx}.json"
    new_address = f"fact_triplets_raw/{new}/{new}_{idx}.json"
    output_dir = f"fact_triplets_raw/{old}_{new}/{mode}/{mode}_id/{mode}_{idx}_id.json"
    with open(old_address, "r") as read_json_file_1:
        previous_python = json.load(read_json_file_1)

    with open(new_address, "r") as read_json_file_2:
        present_python = json.load(read_json_file_2)

    if mode == "unchanged":
        unchanged_relation = []
        for entity in previous_python:
            if entity in present_python:
                small = []
                for first_relation in previous_python[entity]:
                    small.append(first_relation)
                for second_relation in present_python[entity]:
                    if second_relation in small:
                        unchanged_relation.append([entity] + second_relation + ["unchanged"])
 
        with open(output_dir, "w") as write_json_file:
            json.dump(unchanged_relation, write_json_file, indent=4)

    else:
        new_relation = []
        for entity in previous_python:
            if entity in present_python:
                small = []
                for first_relation in previous_python[entity]:
                    if first_relation[0] not in small:
                        small.append(first_relation[0])
                for second_relation in present_python[entity]:
                    if second_relation[0] not in small:
                        if [entity, second_relation[0], second_relation[1]] not in new_relation:
                            new_relation.append([entity, second_relation[0], second_relation[1]])

        updated_relation = []
        for entity in previous_python:
            if entity in present_python:
                small = []
                new_rel = []
                for first_relation in previous_python[entity]:
                    if first_relation[0] not in small:
                        small.append(first_relation[0])
                for second_relation in present_python[entity]:
                    if second_relation[0] not in small:
                        new_rel.append(second_relation[0])
                same = []
                for i in previous_python[entity]:
                    for j in present_python[entity]:
                        if i == j:
                            same.append(i)
                new_prev = []
                new_pres = []
                for i in previous_python[entity]:
                    if i not in same:
                        new_prev.append(i)
                for i in present_python[entity]:
                    if i not in same:
                        if i[0] not in new_rel:
                            new_pres.append(i)
                included = []
                for first_relation in new_prev:
                    relation = first_relation[0]
                    object = first_relation[1]
                    if len(object) > 15:
                        continue
                    for second_relation in new_pres:
                        if relation == second_relation[0]:
                            if len(second_relation[1]) > 15:
                                continue
                            if object != second_relation[1]:
                                updated_relation.append([entity] + first_relation + [second_relation[1]])
                                included.append(second_relation)
                for i in new_pres:
                    if i not in included:
                        updated_relation.append([entity] + i)

        changed_relation = []
        changed_dic = {}
        for new in new_relation:
            sentence = new[0] + new[1] + new[2]
            if sentence not in changed_dic:
                changed_dic[sentence] = 1
                changed_relation.append(new + ["new"])
        for update in updated_relation:
            changed_relation.append(update + ["update"])
        
        with open(output_dir, "w") as write_json_file:
            json.dump(changed_relation, write_json_file, indent=4)     


def fetch_entity(sub, entity_dict, mode, client):
    if sub in entity_dict:
        return sub, entity_dict[sub], 0
    try:
        if mode == "unchanged":
            raise Exception
        entity = client.get(sub, load=True)
        n_requests = 1
    except:
        return sub, None, 0

    l = str(entity)[24:-2]
    try:
        a, b = l.split()
        b = b[1:]
        entity_dict[a] = b
        return sub, b, n_requests
    except:
        a = l.split()[0]
        b = l.split()[1:]
        s = ""
        for j in b:
            s += j + " "
        s = s[1:-1]
        entity_dict[a] = s
        return sub, s, n_requests


def name_parallel(old, new, idx, mode, debug=False, num_workers=10, batch_size=10000):
    id_dir = f"fact_triplets_raw/{old}_{new}/{mode}/{mode}_id/{mode}_{idx}_id.json"
    item_dir = f"fact_triplets_raw/{old}_{new}/{mode}/{mode}_item/{mode}_{idx}_item.json"
    entity_dir = f"fact_triplets_raw/entity_dict.json"

    print(f'Load file from {id_dir}')
    with open(id_dir, "r") as read_json_file_1:
        id_list = json.load(read_json_file_1)
    print(f'Length of id_list: {len(id_list)}')
    with open(f"Property_string.json", "r") as read_dictionary:
        property_dict = json.load(read_dictionary)
    try:
        with open(entity_dir, "r") as read_entity_file:
            entity_dict = json.load(read_entity_file)
    except FileNotFoundError:
        print("No entity_dict file")
        entity_dict = {}
    print(f'Length of entity_dict: {len(entity_dict)}')

    if mode == "changed":
        print('Loading wikidata client...')
        from wikidata.client import Client
        client = Client()
    else:
        client = None

    big_list = []
    new_id = []
    n_requests = 0

    print(f'Processing {mode} data of {old} and {new} with idx {idx}...')
    batched_id_list = [id_list[i*batch_size:min((i+1)*batch_size, len(id_list))] for i in range(len(id_list)//batch_size+1)]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        progress = 0
        processed = 0
        start = time.time()
        for batch in batched_id_list:
            futures = []
            for i in batch:
                if len(i) <= 4:
                    futures.append(executor.submit(fetch_entity, i[0], entity_dict, mode, client))
                    futures.append(executor.submit(fetch_entity, i[2], entity_dict, mode, client))
                elif len(i) == 5:
                    futures.append(executor.submit(fetch_entity, i[0], entity_dict, mode, client))
                    futures.append(executor.submit(fetch_entity, i[2], entity_dict, mode, client))
                    futures.append(executor.submit(fetch_entity, i[3], entity_dict, mode, client))

            for future in as_completed(futures):
                result = future.result()
                progress += 1
                if progress % 10000 == 0:
                    print(f'{progress}/{len(id_list)*2} processed with {n_requests} requests in {time.time()-start:.2f} seconds (estimated time: {(len(id_list)*2-progress)/(progress/(time.time()-start)):.2f} seconds left)')
                if result is None:
                    continue
                sub, entity_name, reqs = result
                n_requests += reqs
                if entity_name:
                    entity_dict[sub] = entity_name
                processed += 1

            if mode == "changed":
                with open(f"fact_triplets_raw/entity_dict_{idx}.json", "w") as write_json_file_3:
                    json.dump(entity_dict, write_json_file_3, indent=4)

    for i in id_list:
        if len(i) <= 4:
            sub, rel, obj, tag = i[0], i[1], i[2], i[3]
            if sub in entity_dict and obj in entity_dict:
                a1, a3 = entity_dict[sub], entity_dict[obj]
                if rel in property_dict:
                    a2 = property_dict[rel]
                    if len(a3.split()) <= 5:
                        big_list.append([a1, a2, a3, tag])
                        new_id.append(i)

        elif len(i) == 5:
            sub, rel, obj, obj2, tag = i[0], i[1], i[2], i[3], i[4]
            if sub in entity_dict and obj in entity_dict and obj2 in entity_dict:
                a1, a3, a4 = entity_dict[sub], entity_dict[obj], entity_dict[obj2]
                if rel in property_dict and len(a3.split()) <= 5 and len(a4.split()) <= 5:
                    a2 = property_dict[rel]
                    if a3.lower() != a4.lower():
                        big_list.append([a1, a2, a4, tag])
                        new_id.append([i[0], i[1], i[3]])

    if not debug:
        with open(item_dir, "w") as write_json_file:
            json.dump(big_list, write_json_file, indent=4)

        with open(id_dir, "w") as write_json_file_2:
            json.dump(new_id, write_json_file_2, indent=4)

    print(f'Number of requests: {n_requests}')
    print(f'Number of processed entities: {processed}')


def merge(old, new, mode):

    big_id_list = []
    big_item_list = []
    with open('fact_triplets_raw/entity_dict.json', "r") as read_entity_file:
        entity_dict = json.load(read_entity_file)
    for i in range(100):
        s = str(i)
        id_dir = f"fact_triplets_raw/{old}_{new}/{mode}/{mode}_id/{mode}_{s}_id.json"
        item_dir = f"fact_triplets_raw/{old}_{new}/{mode}/{mode}_item/{mode}_{s}_item.json"
        ent_dict_name = f"fact_triplets_raw/entity_dict_{s}.json"
        try:
            with open(id_dir, "r") as read_json_1:
                id_list = json.load(read_json_1)
            with open(item_dir, "r") as read_json_2:
                item_list = json.load(read_json_2)
            try:
                with open(ent_dict_name, "r") as read_json_3:
                    ent_dict = json.load(read_json_3)
                entity_dict.update(ent_dict)
            except FileNotFoundError:
                pass
            if len(id_list) != len(item_list):
                continue
            for k in id_list:
                big_id_list.append(k)
            for j in item_list:
                big_item_list.append(j)

            os.remove(ent_dict_name)

        except:
            continue

    print(f'Length of big_id_list: {len(big_id_list)}')
    id_fname = f"fact_triplets_raw/{old}_{new}/{mode}/total_{mode}_id.json"
    item_fname = f"fact_triplets_raw/{old}_{new}/{mode}/total_{mode}_item.json"
    with open(id_fname, "w") as write_json_file_1:
        json.dump(big_id_list, write_json_file_1, indent=4) 
    with open(item_fname, "w") as write_json_file_2:
        json.dump(big_item_list, write_json_file_2, indent=4)
    with open('fact_triplets_raw/entity_dict.json', "w") as write_entity_file:
        json.dump(entity_dict, write_entity_file, indent=4)


def main():
    arg = construct_generation_args()

    mode = arg.mode # mode : unchanged / updated / new
    if mode not in SUPPORT_MODE:
        print(f"Mode {mode} not supported!")
        exit()

    old = arg.old # old : year + month + date, e.g. 20210801
    new = arg.new # new : year + month + date, e.g. 20210901
    idx = arg.idx # idx : One number between 0-100 (Preprocessing is held in every million entities of Wikidata)
    combine = arg.combine # combine : 0 (Not combining created sets by idx) / 1 (Combine all the sets to one json file)

    path = f"fact_triplets_raw/"
    try:
        os.makedirs(path+old, exist_ok=False)
    except:
        pass
    try:
        os.makedirs(path+new, exist_ok=False)
    except:
        pass
    try:
        os.makedirs(path+old+"_"+new, exist_ok=False)
        os.makedirs(path+old+"_"+new+"/changed", exist_ok=False)
        os.makedirs(path+old+"_"+new+"/changed/changed_id", exist_ok=False)
        os.makedirs(path+old+"_"+new+"/changed/changed_item", exist_ok=False)
        os.makedirs(path+old+"_"+new+"/unchanged", exist_ok=False)
        os.makedirs(path+old+"_"+new+"/unchanged/unchanged_id", exist_ok=False)
        os.makedirs(path+old+"_"+new+"/unchanged/unchanged_item", exist_ok=False)
    except:
        pass

    if idx != -1:
        if not arg.debug:
            print(f'Extracting {old}')
            extraction(old, idx) # Extract Wikidata id of previous month
            print(f'Extracting {new}')
            extraction(new, idx) # Extract Wikidata id of new month
            print(f'Filter Unchanged, Updated or New factual instances by id')
            id(old, new, idx, mode) # Filter Unchanged, Updated or New factual instances by id
        print(f'Get name of the entities')
        name_parallel(old, new, idx, mode, debug=arg.debug, num_workers=arg.num_workers, batch_size=arg.batch_size) # Mapping id to string item by using 'WikiMapper'

    if combine == 1:
        merge(old, new, mode) 


if __name__ == '__main__':
    main()
