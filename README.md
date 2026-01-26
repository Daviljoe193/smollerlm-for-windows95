# smollerlm-for-windows95
Just a dumb attempt at vibe coding my way into making an LLM runable on Windows 95 with a Pentium 3.

---

This is just a personal project, to modify the Llama2.c code enough to kinda-sorta recreate Ollama, but for Windows 95.

To run it, first you'll need Ubuntu 24.04 LTS or later (Or earlier, I guess? This just what I got.), and to

```
git clone https://github.com/Daviljoe193/smollerlm-for-windows95
```

Then compile it with SSE support (For Windows 9x, I'd recommend [JHRobotics/simd95](https://github.com/JHRobotics/simd95), the speed boost is worth it) using

```
i686-w64-mingw32-gcc run-smol.c -o run_smol.exe -O3 -march=pentium3 -mtune=pentium3 -mfpmath=sse -msse -funroll-loops -static -s -D_WIN32_WINNT=0x0400 -D__USE_MINGW_ANSI_STDIO=0 -Wno-unknown-pragmas -Wno-attributes -fno-asynchronous-unwind-tables -Wl,--subsystem,console:4.0 -Wl,--allow-multiple-definition -Wl,--wrap=AddVectoredExceptionHandler -Wl,--wrap=RemoveVectoredExceptionHandler -Wl,--wrap=SetThreadStackGuarantee
```

or without SSE support (Will run on slightly older processors than the Pentium 3 this way) using

```
i686-w64-mingw32-gcc run-smol.c -o run_smol.exe -O3 -march=i686 -mtune=pentium3 -mfpmath=387 -mno-sse -mno-sse2 -mno-mmx -static -s -D_WIN32_WINNT=0x0400 -D__USE_MINGW_ANSI_STDIO=0 -Wno-unknown-pragmas -Wno-attributes -fno-asynchronous-unwind-tables -Wl,--subsystem,console:4.0 -Wl,--allow-multiple-definition -Wl,--wrap=AddVectoredExceptionHandler -Wl,--wrap=RemoveVectoredExceptionHandler -Wl,--wrap=SetThreadStackGuarantee
```

Afterwards, you need an LLM and a tokenizer. Currently the scope of this project is so small that it only somewhat supports the SmollerLM family of LLMs by mehmetkeremturkcan on HuggingFace, and no other models currently work. Choose one of his models in that family (I personally went with [this 10 million parameter one](https://huggingface.co/mehmetkeremturkcan/SmollerLM2-10M-sftb), [this 20M one](https://huggingface.co/mehmetkeremturkcan/SmollerLM-20M-Instruct-PrunedPostTrained-sft2), [and this 48M one](https://huggingface.co/mehmetkeremturkcan/SmollerLM-48M-Instruct-ft-sft)), then...

```
pip install -r requirements.txt # Install it in a venv, silly. :3
```

```
python export-smol.py smollerlm2_20m_q80.bin --hf mehmetkeremturkcan/SmollerLM2-10M-sftb
```

```
python export_tokenizer.py --hf mehmetkeremturkcan/SmollerLM2-10M-sftb -o smoller_tokenizer.bin
```

Now you'll have the model and tokenizer in a llama2.c-ish INT8 format. Next, put the tokenizer, model and executable onto a Windows 95 machine, making sure it has at least 64 megabytes of ram.

Finally on Windows 95, you can run it with something like

```
run_smol.exe smollerlm2_10m_q80.bin -n 256 -z smoller_tokenizer.bin -m chat
```

and interact with it more or less like you would in Ollama. It will take several seconds to a minute to load, and doesn't have a proper indicator of if you pressed enter... and also has a broken TUI "scrollbar" that I haven't fixed, requiring you to use PageUp and PageDown to scroll through the chat history. But otherwise, this is a real LLM that can run on really era-inappropriate hardware/software!

Since I ended up losing the CLI flags along the way, they're more or less the same as llama2.c's.

```
Options:
  -t <float>  temperature in [0,inf], default 1.0
  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9
  -s <int>    random seed, default time(NULL)
  -n <int>    number of steps to run for, default 256. 0 = max_seq_len
  -i <string> input prompt
  -z <string> optional path to custom tokenizer
  -m <string> mode: generate|chat, default: generate
  -y <string> (optional) system prompt in chat mode
```

I'm not sure how often I'll update this project, but I hope anyone who finds this has fun! :D
