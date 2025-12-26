import dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm

dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

class Question(BaseModel):
    question: str = Field(description="A pergunta reescrita a partir do texto apresentado pelo usuário")

prompter = llm.with_structured_output(Question)

def generate_question(ementa: str) -> Question:
    return prompter.invoke([
        SystemMessage(content="Sua tarefa é reescrever o texto apresentado pelo usuário em formato de pergunta que possa ser respondida pelo sistema de busca. Quando o texto citar súmulas, não citar diretamente o número da súmula, pois só há interesse no conteúdo (textual) da ementa."),
        HumanMessage(content=ementa)
    ])



if __name__ == "__main__":
    # ementa = """
    # A sanção de declaração de inidoneidade para participar de licitação na Administração Pública Federal (art. 46 da Lei 8.443/1992) pode ser aplicada em razão de fraudes praticadas em processos de dispensa de licitação.
    # """
    # question = generate_question(ementa)
    # print(question)
    # print(question.question)
    path = "jua-dataset/queries.jsonl"
    new_path = "jua-dataset/queries_with_questions.jsonl"
    current_count = len(open(new_path, "r").readlines())
    with open(path, "r") as f:
        lines = f.readlines()
        
        for line in tqdm(lines[current_count:]):
            data = json.loads(line)
            ementa = data["text"]
            question = generate_question(ementa).question
            data['text'] = question
            
            with open(new_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")