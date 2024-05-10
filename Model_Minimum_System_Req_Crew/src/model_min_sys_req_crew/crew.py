from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_groq import ChatGroq

@CrewBase
class MLSystemDesignEngineerCrew():
    """ML System Design Crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    def __init__(self) -> None:
        self.groq_llm = ChatGroq(temperature = 0, model_name = "mixtral-8x7b-32768")

    @agent
    def research_engineer(self) -> Agent:
        return Agent(
            config = self.agents_config['Senior_Research_Engineer'],
            llm = self.groq_llm
        )

    @agent
    def system_design_engineer(self) -> Agent:
        return Agent(
            config = self.agents_config['System_Design_Engineer'],
            llm = self.groq_llm
        )

    @task 
    def research_engineer_task(self) -> Task:
        return Task(
            config = self.tasks_config['Research_Engineer_task'],
            agent = self.research_engineer()
        )

    @task 
    def system_design_engineer_task(self) -> Task:
        return Task(
            config = self.tasks_config['System_Design_Engineer_task'],
            agent = self.system_design_engineer()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents = self.agents,
            tasks = self.tasks,
            process = Process.sequential,
            verbose = 2
        )