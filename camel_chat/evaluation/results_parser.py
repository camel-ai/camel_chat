import argparse
import json
import os


def results_parser(model_name):

    dir_name = f"{model_name}-results"

    topic = "arc_challenge"

    outfile = os.path.join(dir_name, f"{model_name}-{topic}-25-shots.json")
    if os.path.isfile(outfile):
        with open(outfile) as f:
            parsed_json = json.load(f)
        print(parsed_json["results"][topic]["acc_norm"]*100.0)
    else:
        print(f"{outfile} does not exist")

    topic = "hellaswag"

    outfile = os.path.join(dir_name, f"{model_name}-{topic}-10-shots.json")

    if os.path.isfile(outfile):
        with open(outfile) as f:
            parsed_json = json.load(f)
        print(parsed_json["results"][topic]["acc_norm"]*100.0)
    else:
        print(f"{outfile} does not exist")

    TOPICS = ["hendrycksTest-abstract_algebra", "hendrycksTest-anatomy", "hendrycksTest-astronomy", "hendrycksTest-business_ethics", "hendrycksTest-clinical_knowledge", "hendrycksTest-college_biology", "hendrycksTest-college_chemistry", "hendrycksTest-college_computer_science", "hendrycksTest-college_mathematics", "hendrycksTest-college_medicine", "hendrycksTest-college_physics", "hendrycksTest-computer_security", "hendrycksTest-conceptual_physics", "hendrycksTest-econometrics", "hendrycksTest-electrical_engineering", "hendrycksTest-elementary_mathematics", "hendrycksTest-formal_logic", "hendrycksTest-global_facts", "hendrycksTest-high_school_biology", "hendrycksTest-high_school_chemistry", "hendrycksTest-high_school_computer_science", "hendrycksTest-high_school_european_history", "hendrycksTest-high_school_geography", "hendrycksTest-high_school_government_and_politics", "hendrycksTest-high_school_macroeconomics", "hendrycksTest-high_school_mathematics", "hendrycksTest-high_school_microeconomics", "hendrycksTest-high_school_physics", "hendrycksTest-high_school_psychology", "hendrycksTest-high_school_statistics", "hendrycksTest-high_school_us_history", "hendrycksTest-high_school_world_history", "hendrycksTest-human_aging", "hendrycksTest-human_sexuality", "hendrycksTest-international_law", "hendrycksTest-jurisprudence", "hendrycksTest-logical_fallacies", "hendrycksTest-machine_learning", "hendrycksTest-management", "hendrycksTest-marketing", "hendrycksTest-medical_genetics", "hendrycksTest-miscellaneous", "hendrycksTest-moral_disputes", "hendrycksTest-moral_scenarios", "hendrycksTest-nutrition", "hendrycksTest-philosophy", "hendrycksTest-prehistory", "hendrycksTest-professional_accounting", "hendrycksTest-professional_law", "hendrycksTest-professional_medicine", "hendrycksTest-professional_psychology", "hendrycksTest-public_relations", "hendrycksTest-security_studies", "hendrycksTest-sociology", "hendrycksTest-us_foreign_policy", "hendrycksTest-virology", "hendrycksTest-world_religions"]

    hendrycksTest = []

    for topic in TOPICS:
        outfile = os.path.join(dir_name, f"{model_name}-{topic}-5-shots.json")
        if os.path.isfile(outfile):
            with open(outfile) as f:
                parsed_json = json.load(f)
            acc_norm = parsed_json["results"][topic]["acc_norm"]*100.0
            print(acc_norm)
            hendrycksTest.append(acc_norm)
        else:
            print(f"{outfile} does not exist")

    print("")

    topic = "truthfulqa_mc"

    outfile = os.path.join(dir_name, f"{model_name}-{topic}-0-shots.json")

    if os.path.isfile(outfile):
        with open(outfile) as f:
            parsed_json = json.load(f)
        print(parsed_json["results"][topic]["mc2"]*100.0)
    else:
        print(f"{outfile} does not exist")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model specified in eval_tasks.sh")
    args = parser.parse_args()
    
    results_parser(args.model_name)


