# smollerlm-for-~~windows95~~macos9x
Just a dumb attempt at vibe coding my way into making an LLM runable on ~~Windows 95 with a Pentium 3 (And other similarly unreasonable targets, now even supporting Pentium II and lower)~~ Mac OS 8.6 and 9.x.

Yep, another branch, another ridiculous port.

![](https://raw.githubusercontent.com/Daviljoe193/smollerlm-for-windows95/refs/heads/macos-classic/llamac-smol-ghdemo-macclassicenv.avif)
<sup>I really need to get my G4 iMac out of storage to get a real feel for performance.</sup>

---

This is just a personal project, to modify the Llama2.c code enough to kinda-sorta recreate Ollama, but for Windows 95, and now also Mac OS 8.6 and 9.x.

If you don't want to build everything yourself, I have a prebuild in the releases section, [here](https://github.com/Daviljoe193/smollerlm-for-windows95/releases). Otherwise...

---

To run it, first you'll need Ubuntu 24.04 LTS or later (Or earlier, I guess? This just what I got.) for making the model/tokenizer files, and to

```
git clone -b macos-classic https://github.com/Daviljoe193/smollerlm-for-windows95
```

You'll also need a G4 Mac running either Mac OS 8.6 to 9.x, or Mac OS 10.x on G4 (Or `qemu-system-ppc -cpu G4` type setup) with a classic environment set up. On that environment, you'll need to install Metrowerks CodeWarrior Pro 5 (Which, pst, is abandonware that's on MacintoshRepository) with all the Mac components for the IDE installed. Now you'll need a directory structure kinda like this...

```
smollerlm-for-macos9x/
├── RunSmol.c
└── RunSmol.r
```

Now you can open the IDE, and choose to `Import Project...` the `smollerlm-for-macos8x.xml` file. You'll then save it inside the same directory structure you created above, with whatever project name you please, and like that, you can compile. Just hit the button that looks like a cross between the `📝` and `✍️` emoji, and you can then immediatly quit CodeWarrior! But you're not done!!!

Afterwards, you need an LLM and a tokenizer. Currently the scope of this project is so small that it only somewhat supports the SmollerLM family of LLMs by mehmetkeremturkcan on HuggingFace, and no other models currently work. Back on the Ubuntu machine, choose one of his models in that family (I personally went with [this 10 million parameter one](https://huggingface.co/mehmetkeremturkcan/SmollerLM2-10M-sftb), [this 20M one](https://huggingface.co/mehmetkeremturkcan/SmollerLM-20M-Instruct-PrunedPostTrained-sft2), [and this 48M one](https://huggingface.co/mehmetkeremturkcan/SmollerLM-48M-Instruct-ft-sft)), then...

```
pip install -r requirements.txt # Install it in a venv, silly. :3
```

```
python export-smol.py smollerlm2_10m_q80.bin --hf mehmetkeremturkcan/SmollerLM2-10M-sftb
```

```
python export_tokenizer.py --hf mehmetkeremturkcan/SmollerLM2-10M-sftb -o tokenizer.bin
```

Now you'll have the model and tokenizer in a llama2.c-ish INT8 format. Next, put the tokenizer and model making sure it has at least 128 megabytes of ram (I'm optimizing still, so this is bloated compared to the final intended requirements). The model can go anywhere, really. But the `tokenizer.bin` must be in the same directory as the compiled app.

When you first launch the app, it'll want to know what model you're using first and foremost.

![](https://raw.githubusercontent.com/Daviljoe193/smollerlm-for-windows95/refs/heads/macos-classic/startup-error.png)

![](https://raw.githubusercontent.com/Daviljoe193/smollerlm-for-windows95/refs/heads/macos-classic/startup-errorp2.png)

Now you can get onto choosing anything else you'd like prior to starting the chat/oneoff-inference.

![](https://raw.githubusercontent.com/Daviljoe193/smollerlm-for-windows95/refs/heads/macos-classic/startup-errorp3.png)

Then finally click Start to open up a "terminal" with the chat interface. Here you can either interact with it like any other chat-ready LLM, or enter `/bye` to gracefully end chat. Currently a lot of the `/` commands aren't implemented, and are very much a priority to re-add. Also scrollback is weird on the "CodeWarrior Terminal", since the version I'm using might crash after 32k characters are added, but that's another issue for another day. 

![](https://raw.githubusercontent.com/Daviljoe193/smollerlm-for-windows95/refs/heads/macos-classic/startup-errorp4.png)

If the "terminal" closes prematurely after printing out the `----------------------------` bit, then that means your `tokenizer.bin` isn't in the same directory as the compiled app.

I'm not sure how often I'll update this project, but I hope anyone who finds this has fun! :D

---

### Inspiration

Inspiration came from messing around with small LLMs (I do this a lot, since it's free), then coming across [mehmetkeremturkcan](https://huggingface.co/mehmetkeremturkcan)'s unusually small LLMs that had an abnormally SmolLM-like chat template. Converting his 10M model to q_8 GGUFs, I noticed that the model size was so absurdly small that it'd fit onto 10 floppy disks. Then I though Windows 95 running an LLM like this must've been possible.

One thing let to another, and I spent several days making that stupid joke of an idea real. Then I thought "What about the `Yikes!` G4 Mac with the original OS 8.6 install?" That's it, that's the depth of the story. This was never meant to be a serious project, but it was a fun one, and worth it.
