import torch
import threading
import time
import tqdm
import itertools
import pandas as pd
from PIL import ImageFile
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

ImageFile.LOAD_TRUNCATED_IMAGES = True

def thread_process(data, thread_id, results, progress_bar, lock):
    print(f"Thread number {thread_id} working on data from {data.index[0]} to {data.index[-1]}")

    model_path = "deepseek-ai/Janus-Pro-1B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, use_fast=True)
    tokenizer = vl_chat_processor.tokenizer

    device = torch.device(f"cuda:{thread_id%torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).to(device).eval()

    answers = []
    for filename in data.values:

        question = f"What is the scene depicted in this painting? Describe also the subjects within, trying to identify them if possible. If it can help you extract additional information, the file name is '{filename.split('/')[-1]}'. Do not include the file name in your response, and your descriptions should be divided in paragraphs, without using lists. The response should be around 200 words long."
        image = f"../../WikiArt/wikiart/{filename}"


        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]


        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
            #prompt=f"<image_placeholder>\n{question}", images=pil_images, force_batchify=True
        ).to(vl_gpt.device)
        #print(type(prepare_inputs))
        # # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        with torch.no_grad():
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )

            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        answers.append((filename, answer.replace('\n', ' ')))
        with lock:
            progress_bar.update(1)
            progress_bar.refresh()
    results[thread_id] = answers

wikiart_file = pd.concat([
    pd.read_csv(f"../../splits/wikiart_{split}.csv")
 for split in ["train", "val", "test"]]).reset_index()

wikiart_file = wikiart_file.iloc[32497:]

n_threads = 2
split_len = len(wikiart_file)//n_threads
splits = [wikiart_file["path"][idx*split_len:(idx+1)*split_len] for idx in range(n_threads-1)]
splits.append(wikiart_file["path"][(n_threads-1)*split_len:])

start_time = time.time()

results = {}
threads = []
lock = threading.Lock()
with tqdm.tqdm(total=len(wikiart_file)) as progress_bar:
    for i in range(n_threads):
        thread = threading.Thread(target=thread_process, args=(splits[i], i, results, progress_bar, lock))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

wikiart_described = pd.DataFrame(data=itertools.chain.from_iterable(results.values()), columns=["path", "description"])
wikiart_described.to_csv("wikiart_described_2.csv", sep=';')
end_time = time.time()

#print(results)
print("Done")
print(f"Elapsed time: {end_time - start_time}")