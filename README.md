# smollerlm-for-windows95
Just a dumb attempt at vibe coding my way into making an LLM runable on Windows 95 with a Pentium 3 (And other similarly unreasonable targets, now even supporting Pentium II and lower).

SPECIFICALLY, this is the testing branch for getting older support working. I broke some things on my way here, so for Pentium II and III + PPX OS X support, use the main branch. Expect this to be properly merged in when I'm able to. Though I'll bundle it all into the releases. Also the seeds are weird on the Open Watcom builds, so don't expect the same outputs from the mingw builds vs the Watcom builds, though that'll also be fixed soon.

![](https://raw.githubusercontent.com/Daviljoe193/smollerlm-for-windows95/refs/heads/main/llamac-smol-ghdemo.avif)

---

This is just a personal project, to modify the Llama2.c code enough to kinda-sorta recreate Ollama, but for Windows 95.

If you don't want to build everything yourself, I have a prebuild in the releases section, [here](https://github.com/Daviljoe193/smollerlm-for-windows95/releases). Otherwise...

---

To run it, first you'll need Ubuntu 24.04 LTS or later (Or earlier, I guess? This just what I got.), and to

```
git clone -b Open-Watcom https://github.com/Daviljoe193/smollerlm-for-windows95
```

and install Open Watcom. Since there's no DEB for it that I know of (Only a crappy SNAP), I've made a .deb myself. Download it from the releases page, and install it with

```
sudo apt install ./openwatcom-v2_2.0-20251103_amd64.deb
```

Also in the terminal you're compiling in, make sure to export the following...

```
export WATCOM=/opt/watcom
export INCLUDE=$WATCOM/h:$WATCOM/h/nt
export LIB=$WATCOM/lib386:$WATCOM/lib386/nt
export PATH=$WATCOM/binl:$PATH
```

Now for Pentium MMX / AMD K6 / K6_2, you'll build with

```
wcl386 run-smol.c -fe=run_smol.exe -l=nt -bt=nt -5 -fp5 -otexan -s -d_WIN32_WINNT=0x0400
```

...or for Intel 486DX / Pentium (Scalar only. Works, but this is the masochist route!)

```
wcl386 run-smol.c -fe=run_smol.exe -l=nt -bt=nt -4 -fp3 -otexan -s -d_WIN32_WINNT=0x0400
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

Now you'll have the model and tokenizer in a llama2.c-ish INT8 format. Next, put the tokenizer, model and executable onto a Windows 95 machine, making sure it has at least 64 megabytes of ram.

Finally on Windows 95, you can install `msvcrt.dll` (Yes, you need this), then finally once you have that, you can run your LLM with something like

```
run_smol.exe smollerlm2_10m_q80.bin -z smoller_tokenizer.bin
```

Or on PPC Mac OS X with

```
./run_smol smollerlm2_10m_q80.bin -z smoller_tokenizer.bin
```

and interact with it more or less like you would in Ollama. It will take several seconds to a minute to load, and doesn't have a proper indicator of if you pressed enter... and also has a broken TUI "scrollbar" that I haven't fixed, requiring you to use PageUp and PageDown to scroll through the chat history. But otherwise, this is a real LLM that can run on really era-inappropriate hardware/software!

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
