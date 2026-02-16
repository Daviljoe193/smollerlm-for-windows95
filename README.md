# smollerlm-for-windows95
Just a dumb attempt at vibe coding my way into making an LLM runable on four stock Atari Falcons with 14 megabytes of ram.

![](https://raw.githubusercontent.com/Daviljoe193/smollerlm-for-windows95/refs/heads/atari-falcon-dist/itshouldn'twork.png)

---

This is just a personal project, to modify the Llama2.c code enough to kinda-sorta recreate Ollama, but for ~~Windows 95~~ EVERYTHING.

If you don't want to build everything yourself, I have a bunch of convenient prebuilds for every platform I've worked on in the releases section, [here](https://github.com/Daviljoe193/smollerlm-for-windows95/releases).

---

## Port checklist

- [x] Windows 95 Pentium II and newer
- [x] Windows 95 386DX /w FPU and newer
- [x] Mac OS 9 on G4
- [x] Amiga Workbench 3.1 (Needs speed boost)
- [ ] BeOS
- [ ] RISC OS StrongARM and newer (This one's genuinely killing me at the moment, do NOT do this unless you hate fun)
- [x] Atari Falcon

---
Onto the build instructions.

---

To run it, first you'll need Ubuntu 24.04 LTS or later (Or earlier, I guess? This just what I got.), and to

```
git clone -b atari-falcon-dist https://github.com/Daviljoe193/smollerlm-for-windows95
```

and also for compiling the Falcon port

```
sudo add-apt-repository ppa:vriviere/ppa
sudo apt update
sudo apt install cross-mint-essential
```

Then simply compile it using

```
m68k-atari-mint-gcc run-smol.c -o smol30.tos -O3 -m68030 -m68881 -mhard-float -fomit-frame-pointer -funroll-loops -ffast-math -lm
```

Afterwards, you need an LLM and a tokenizer. Currently the scope of this project is so small that it only somewhat supports the SmollerLM family of LLMs by mehmetkeremturkcan on HuggingFace, and no other models currently work. Choose one of his models in that family (I personally went with [this 10 million parameter one](https://huggingface.co/mehmetkeremturkcan/SmollerLM2-10M-sftb), [this 20M one](https://huggingface.co/mehmetkeremturkcan/SmollerLM-20M-Instruct-PrunedPostTrained-sft2), [and this 48M one](https://huggingface.co/mehmetkeremturkcan/SmollerLM-48M-Instruct-ft-sft)) (Currently only the 10M ver has been tested on this port), then...

```
pip install -r requirements.txt # Install it in a venv, silly. :3
```

```
python export-smol.py smollm.bin --hf mehmetkeremturkcan/SmollerLM2-10M-sftb
```

```
python export_tokenizer.py --hf mehmetkeremturkcan/SmollerLM2-10M-sftb -o token.bin
```

Now you'll have the model and tokenizer in a llama2.c-ish INT8 format. Next, put the tokenizer, model and executable onto four Atari Falcons. Make sure they're all connected as such...

- MIDI OUT on unit 1 connected to MIDI IN on unit 2
- MIDI OUT on unit 2 connected to MIDI IN on unit 3
- MIDI OUT on unit 3 connected to MIDI IN on unit 4
- MIDI OUT on unit 4 connected to MIDI IN on unit 1

And that each has the appropriate unit and total count defined in the smol.cfg...

```
node=1
total=4
model=SMOLLM.BIN
tokenizer=TOKEN.BIN
```

And finally, run the program on all the nodes except 1 first, then finally (For real) run it on node 1. Press space on node 1, and hopefully the contents on all four nodes will change (If they don't well... I really would like to run this on real hardware). Now just wait, and eventually a `>>> ` prompt will appear for you to interact with it more or less like you would in Ollama. This is a real LLM that can run on really era-inappropriate hardware/software!

I'm not sure how often I'll update this project, but I hope anyone who finds this has fun! :D

---

### Inspiration

Inspiration came from messing around with small LLMs (I do this a lot, since it's free), then coming across [mehmetkeremturkcan](https://huggingface.co/mehmetkeremturkcan)'s unusually small LLMs that had an abnormally SmolLM-like chat template. Converting his 10M model to q_8 GGUFs, I noticed that the model size was so absurdly small that it'd fit onto 10 floppy disks. Then I though Windows 95 running an LLM like this must've been possible.

One thing let to another, and I spent several days making that stupid joke of an idea real, then did the same for Mac OS 9, then Amiga, and now on Atari's last home computer. That's it, that's the depth of the story. This was never meant to be a serious project, but it was a fun one, and worth it.
