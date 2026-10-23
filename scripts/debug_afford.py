"""Debug: what does the planner decide and what ends up in messages?"""

import json
import random

import psycopg

from agentmem.agent import MemoryAgent, _missing_numbers
from agentmem.db import migrate
from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.settings import get_settings
from agentmem.units import toolbox
from agentmem.units import conversation


def main():
    s = get_settings()
    migrate(s.database_url)
    conn = psycopg.connect(s.database_url, autocommit=True)
    user = f"aff-dbg-{random.randint(10**8, 10**9 - 1)}"
    agent = MemoryAgent(conn, user_id=user, chat=ChatClient(),
                        embedder=Embedder())
    text = ("I make 6,800 a month and my other debt payments are 1,200 "
            "monthly. Under Acme's DTI limit, what's the most I can "
            "qualify to pay per month overall, and what does that leave?")
    thread = "dbg"
    qv = agent.embedder.embed(text)
    seq = conversation.append_turn(conn, user_id=user, thread_id=thread,
                                   role="user", content=text,
                                   embedding=qv)["seq"]

    # replicate the loop with visibility
    from agentmem.agent import SYSTEM_BASE
    found = toolbox.search_tools(conn, qv, k=4)
    menu = "\n".join(
        f"[{t['name']}] {t['description']} params: {json.dumps(t['params'])}"
        for t in found)
    menu += "\n" + "\n".join(
        f"[{n}] params: {json.dumps(sp['params'])} ttl={sp['ttl']}s"
        for n, sp in __import__("agentmem.tools", fromlist=["TOOLS"]).TOOLS.items())

    outcomes = []
    for rnd in range(3):
        plan = agent.chat.complete(
            [
                {"role": "system",
                 "content": toolbox.PLANNER_PROMPT.format(menu=menu)},
                {"role": "user", "content": toolbox.planner_payload(
                    question=text, facts={}, results_so_far=[
                        {"tool": t["name"], "result": json.loads(t["result_for_model"])}
                        for t in outcomes
                    ])},
            ],
            temperature=0.0, max_tokens=200,
            response_format={"type": "json_object"},
        )
        print(f"--- round {rnd}: plan={plan.text}")
        d = json.loads(plan.text)
        if d.get("action") != "tool":
            break
        out = toolbox.run_tool(conn, user_id=user, thread_id=thread,
                               name=d["name"], args=d.get("arguments", {}))
        print(f"    ran {d['name']} args={d.get('arguments')} "
              f"source={out['source']}")
        print(f"    result_for_model={out['result_for_model']}")
        outcomes.append(out)

    print("missing_numbers for 'the answer is $1,860' ->",
          _missing_numbers("The answer is $1,860.", outcomes, text))
    print("missing_numbers for 'you cannot qualify, -$1,200' ->",
          _missing_numbers("you cannot qualify, -$1,200.", outcomes, text))

    for table in ("mem.persona",):
        conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (user,))
    conn.close()


if __name__ == "__main__":
    main()
