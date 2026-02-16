# smollerlm-for-windows95
Just a dumb attempt at vibe coding my way into making an LLM runable on Windows 95 with a Pentium 3 (And other similarly unreasonable targets, now even supporting Pentium II and lower). (Check the branches for [Pentium / Pentium MMX / AMD K6 / K6_2 code](https://github.com/Daviljoe193/smollerlm-for-windows95/tree/Open-Watcom) that hasn't yet been merged), or for [the experimental Mac OS 8.x / 9.x version for the "Yikes!" G4 Mac and other G4 Mac models](https://github.com/Daviljoe193/smollerlm-for-windows95/tree/macos-classic), which is going to be a nightmare to merge codebases with. Even more so is [the new Amiga port](https://github.com/Daviljoe193/smollerlm-for-windows95/tree/amiga-3.1), and [the more freakish distributed Atari Falcon port](https://github.com/Daviljoe193/smollerlm-for-windows95/tree/atari-falcon-dist).

![](https://raw.githubusercontent.com/Daviljoe193/smollerlm-for-windows95/refs/heads/main/llamac-smol-ghdemo.avif)

---

This is just a personal project, to modify the Llama2.c code enough to kinda-sorta recreate Ollama, but for Windows 95.

If you don't want to build everything yourself, I have a bunch of convenient prebuilds for every platform I've worked on in the releases section, [here](https://github.com/Daviljoe193/smollerlm-for-windows95/releases).

---

## Port checklist

- [x] Windows 95 Pentium II and newer
- [x] Windows 95 386DX /w FPU and newer
- [x] Mac OS 9 on G4
- [x] Amiga Workbench 3.1 (Needs speed boost)
- [ ] BeOS
- [ ] RISC OS StrongARM and newer (This one's genuinely killing me at the moment, do NOT do this unless you hate fun)

---
Onto the build instructions.

---

To run it, first you'll need Ubuntu 24.04 LTS or later (Or earlier, I guess? This just what I got.), and to

```
git clone https://github.com/Daviljoe193/smollerlm-for-windows95
```

and also for compiling the Windows 95 versions

```
sudo apt install mingw-w64
```

Then compile it with SSE support (For Windows 9x, I'd recommend [JHRobotics/simd95](https://github.com/JHRobotics/simd95), the speed boost is worth it) using

```
i686-w64-mingw32-gcc run-smol.c -o run_smol.exe -O3 -march=pentium3 -mtune=pentium3 -mfpmath=sse -msse -funroll-loops -static -s -D_WIN32_WINNT=0x0400 -D__USE_MINGW_ANSI_STDIO=0 -Wno-unknown-pragmas -Wno-attributes -fno-asynchronous-unwind-tables -Wl,--subsystem,console:4.0 -Wl,--allow-multiple-definition -Wl,--wrap=AddVectoredExceptionHandler -Wl,--wrap=RemoveVectoredExceptionHandler -Wl,--wrap=SetThreadStackGuarantee
```

...or if you instead want 3DNow! support (Can't leave '90s Team Red high and dry, after all), using **GARBAGE MODEL OUTPUT, WILL FIX**

```
i686-w64-mingw32-gcc run-smol.c -o run_smol.exe -O3 -march=k6-2 -mtune=athlon -m3dnow -fno-math-errno -ffinite-math-only -funsafe-math-optimizations -funroll-loops -static -s -D__3dNOW__ -D_WIN32_WINNT=0x0400 -D__USE_MINGW_ANSI_STDIO=0 -Wno-unknown-pragmas -Wno-attributes -fno-asynchronous-unwind-tables -Wl,--subsystem,console:4.0 -Wl,--allow-multiple-definition -Wl,--wrap=AddVectoredExceptionHandler -Wl,--wrap=RemoveVectoredExceptionHandler -Wl,--wrap=SetThreadStackGuarantee
```

Or for slightly older Team Red, on the original K6 (using MMX) **CURRENTLY BROKEN, WILL FIX**

```
i686-w64-mingw32-gcc run-smol.c -o run_smol.exe -O3 -march=k6 -mtune=k6 -mmmx -mno-3dnow -mno-sse -mno-sse2 -mfpmath=387 -funroll-loops -static -s -D__MMX__ -D_WIN32_WINNT=0x0400 -D__USE_MINGW_ANSI_STDIO=0 -Wno-unknown-pragmas -Wno-attributes -fno-asynchronous-unwind-tables -Wl,--subsystem,console:4.0 -Wl,--allow-multiple-definition -Wl,--wrap=AddVectoredExceptionHandler -Wl,--wrap=RemoveVectoredExceptionHandler -Wl,--wrap=SetThreadStackGuarantee
```

...or for Pentium II with MMX...

```
i686-w64-mingw32-gcc run-smol.c -o run_smol.exe -O3 -march=pentium2 -mtune=pentium2 -mmmx -mno-sse -mno-sse2 -mfpmath=387 -funroll-loops -static -s -D__MMX__ -D_WIN32_WINNT=0x0400 -D__USE_MINGW_ANSI_STDIO=0 -Wno-unknown-pragmas -Wno-attributes -fno-asynchronous-unwind-tables -Wl,--subsystem,console:4.0 -Wl,--allow-multiple-definition -Wl,--wrap=AddVectoredExceptionHandler -Wl,--wrap=RemoveVectoredExceptionHandler -Wl,--wrap=SetThreadStackGuarantee
```

...or Pentium MMX... **CURRENTLY BROKEN, WILL FIX**

```
i686-w64-mingw32-gcc run-smol.c -o run_smol.exe -O3 -march=pentium-mmx -mtune=pentium-mmx -mmmx -mno-sse -mno-sse2 -mfpmath=387 -funroll-loops -static -s -D__MMX__ -D_WIN32_WINNT=0x0400 -D__USE_MINGW_ANSI_STDIO=0 -Wno-unknown-pragmas -Wno-attributes -fno-asynchronous-unwind-tables -Wl,--subsystem,console:4.0 -Wl,--allow-multiple-definition -Wl,--wrap=AddVectoredExceptionHandler -Wl,--wrap=RemoveVectoredExceptionHandler -Wl,--wrap=SetThreadStackGuarantee
```

...or for Intel 486DX (Scalar only, for masochists) **CURRENTLY BROKEN, WILL FIX**

```
i686-w64-mingw32-gcc run-smol.c -o run_smol.exe -O3 -march=i486 -mtune=i486 -mno-mmx -mno-sse -mno-sse2 -mfpmath=387 -funroll-loops -static -s -D_WIN32_WINNT=0x0400 -D__USE_MINGW_ANSI_STDIO=0 -Wno-unknown-pragmas -Wno-attributes -fno-asynchronous-unwind-tables -Wl,--subsystem,console:4.0 -Wl,--allow-multiple-definition -Wl,--wrap=AddVectoredExceptionHandler -Wl,--wrap=RemoveVectoredExceptionHandler -Wl,--wrap=SetThreadStackGuarantee
```

...or for Pentium (Scalar only, for masochists with standards) **ALSO CURRENTLY BROKEN, WILL FIX**

```
i686-w64-mingw32-gcc run-smol.c -o run_smol.exe -O3 -march=pentium -mtune=pentium -mno-mmx -mno-sse -mno-sse2 -mfpmath=387 -funroll-loops -static -s -D_WIN32_WINNT=0x0400 -D__USE_MINGW_ANSI_STDIO=0 -Wno-unknown-pragmas -Wno-attributes -fno-asynchronous-unwind-tables -Wl,--subsystem,console:4.0 -Wl,--allow-multiple-definition -Wl,--wrap=AddVectoredExceptionHandler -Wl,--wrap=RemoveVectoredExceptionHandler -Wl,--wrap=SetThreadStackGuarantee
```

PowerPC G4 support is experimental (and unimpressive ATM, especially since I'm aiming for Yikes! support), and as of now, I cannot figure out how to cross compile this. On a PowerPC Mac with a G4 or better (Or in a QEMU G4 environment), with Xcode 2.5 installed, run

```
gcc run-smol.c -o run_smol -O3 -mcpu=7450 -maltivec -mabi=altivec -lm -D__G4__l
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
