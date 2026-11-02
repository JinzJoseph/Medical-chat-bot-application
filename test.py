from app.agent import graph

config={
    'configurable':{
        "thread_id":"user_123"
    }
}

result=graph.invoke(
    {
        "question":"what causes a headache?",
        "history":[],
        "image_description":"",
        "kb":[],
        "web":[],
        "answer":"",
        "route":""
    },
    config=config
)
print("Final result:",result)