# smollerlm-for-~~windows95~~amiga-3.1
Just a dumb attempt at vibe coding my way into making an LLM runable on ~~Windows 95 with a Pentium 3 (And other similarly unreasonable targets, now even supporting Pentium II and lower)~~ Amiga Workbench 3.1 on a Motorolla 68060.

Yep, another branch, another ridiculous port.

![WILL PUT IMAGE HERE EVENTUALLY.](https://raw.githubusercontent.com/Daviljoe193/smollerlm-for-windows95/refs/heads/amiga-3.1/llamac-smol-ghdemo-amiga-placeholder.png)

###### Placeholder still, will be replaced with proper animaged demo.
---

This is just a personal project, to modify the Llama2.c code enough to kinda-sorta recreate Ollama, but for Windows 95, then Mac OS 8.6 and 9.x, and now for Amiga Workbench 3.1.

If you don't want to build everything yourself, I have a bunch of convenient prebuilds for every platform I've worked on in the releases section, [here](https://github.com/Daviljoe193/smollerlm-for-windows95/releases). Otherwise...

---

To run it, first you'll need Ubuntu 24.04 LTS or later (Or earlier, I guess? This just what I got.) for making the model/tokenizer files, and to

```
git clone -b amiga-3.1 https://github.com/Daviljoe193/smollerlm-for-windows95
```

and also for compiling the Amiga version, you'll need to build `m68k-amigaos-gcc` [from this repository](https://github.com/AmigaPorts/m68k-amigaos-gcc). I can't tl;dr this, so you have to do it on your own.

Then with that compiler ready, compile with

```
m68k-amigaos-gcc run-smol-amiga.c -o run_smol_060 -O3 -m68060 -mhard-float -noixemul -fomit-frame-pointer -funroll-loops -ffast-math -lm
```

Afterwards, you need an LLM and a tokenizer. Currently the scope of this project is so small that it only somewhat supports the SmollerLM family of LLMs by mehmetkeremturkcan on HuggingFace, and no other models currently work. Choose one of his models in that family (I personally went with [this 10 million parameter one](https://huggingface.co/mehmetkeremturkcan/SmollerLM2-10M-sftb), [this 20M one](https://huggingface.co/mehmetkeremturkcan/SmollerLM-20M-Instruct-PrunedPostTrained-sft2), [and this 48M one](https://huggingface.co/mehmetkeremturkcan/SmollerLM-48M-Instruct-ft-sft)), then...

```
pip install -r requirements.txt # Install it in a venv, silly. :3
```

```
python export-smol.py smollerlm2_10m_q80.bin --hf mehmetkeremturkcan/SmollerLM2-10M-sftb
```

```
python export_tokenizer.py --hf mehmetkeremturkcan/SmollerLM2-10M-sftb -o smoller_tokenizer.bin
```

Now you'll have the model and tokenizer in a llama2.c-ish INT8 format. Next, put the tokenizer, model and executable onto an appropriate (With 64 megabytes of added fast ram minimum, 128 mb for the bigger models. Also with a 68060 CPU, the one with a built in FPU) Amiga with Workbench 3.1.

Finally on the Amiga, you can run your LLM with something like (And the first command is important to prevent a crash, even with the needed fast ram)

```
Stack 100000
run_smol smollerlm2_10m_q80.bin -z smoller_tokenizer.bin
```

and interact with it more or less like you would in Ollama. Just know it's VERY slow at the moment, only about double the performance of the 486DX Windows 95 port on a 25 MHz 486DX.

If run without any flags, it'll print out a help message.

```
Usage:   run <checkpoint> [options]
Example: run model.bin -n 256 -i "Once upon a time"
Options:
  -t <float>  temperature in [0,inf], default 1.0
  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9
  -s <int>    random seed, default time(NULL)
  -n <int>    number of steps to run for, default 256. 0 = max_seq_len
  -i <string> input prompt
  -z <string> optional path to custom tokenizer
  -m <string> mode: generate|chat, default: chat
  -y <string> (optional) system prompt in chat mode
```

I'm not sure how often I'll update this project, but I hope anyone who finds this has fun! :D

---

### Inspiration

Inspiration came from messing around with small LLMs (I do this a lot, since it's free), then coming across [mehmetkeremturkcan](https://huggingface.co/mehmetkeremturkcan)'s unusually small LLMs that had an abnormally SmolLM-like chat template. Converting his 10M model to q_8 GGUFs, I noticed that the model size was so absurdly small that it'd fit onto 10 floppy disks. Then I though Windows 95 running an LLM like this must've been possible.

One thing let to another, and I spent several days making that stupid joke of an idea real. That's it, that's the depth of the story. This was never meant to be a serious project, but it was a fun one, and worth it.
