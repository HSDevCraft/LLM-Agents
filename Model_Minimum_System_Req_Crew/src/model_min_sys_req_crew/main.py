import os
from dotenv import load_dotenv
load_dotenv()
from model_min_sys_req_crew.crew import MLSystemDesignEngineerCrew


def run():
    inputs = {
        'model_name' : input("Enter the model name: "),
    }

    MLSystemDesignEngineerCrew().crew().kickoff(inputs = inputs)

if __name__ == "__main__":
    run()