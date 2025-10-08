🧠 Markdown Cheat Sheet (for VS Code)

📝 Basic Text Formatting
Style	Markdown Syntax	Example	Output
Bold	**text** or __text__	**bold text**	bold text
Italic	*text* or _text_	*italic text*	italic text
Bold + Italic	***text***	***both***	both
Strikethrough	~~text~~	~~delete~~	delete
Inline code	`code`	`print()`	print()
Blockquote	> text	> note	> note
Line break	End a line with two spaces	line 1␣␣
line 2	line 1
line 2

🧱 Headings
Level	Syntax	Example
H1	# Heading 1	# Heading 1
H2	## Heading 2	## Heading 2
H3	### Heading 3	### Heading 3
H4	#### Heading 4	#### Heading 4
H5	##### Heading 5	##### Heading 5
H6	###### Heading 6	###### Heading 6

📋 Lists

Unordered list:

- Item 1
  - Subitem 1
  - Subitem 2
* Or use asterisks
+ Or plus signs


Ordered list:

1. First
2. Second
   1. Subitem
   2. Subitem


Task list:

- [x] Done
- [ ] To do


✅ Output:

 Done

 To do

🔗 Links & Images
Type	Syntax	Example	Output
Link	[title](https://url.com)	[OpenAI](https://openai.com)	OpenAI

Image	![alt text](image.png)	![Logo](logo.png)	

Reference Link	[Google][1] + [1]: https://google.com	[Google][1]	[Google][1]
💻 Code Blocks

Inline code:
`code` → code

Multiline / Block code:

<pre> ```language code here ``` </pre>

Example:

def hello():
    print("Hello Markdown!")


VS Code automatically adds syntax highlighting.

📊 Tables
| Name | Age | Job |
|------|-----|-----|
| Alice | 25 | Developer |
| Bob | 30 | Designer |


Output:

Name	Age	Job
Alice	25	Developer
Bob	30	Designer

Align columns:

| Left | Center | Right |
|:-----|:------:|------:|
| a | b | c |

🖼️ Horizontal Line
---
***
___


All produce:

🧩 VS Code Markdown Shortcuts
Action	Shortcut (Windows/Linux)	Shortcut (Mac)
Bold	Ctrl + B	Cmd + B
Italic	Ctrl + I	Cmd + I
Open Preview	Ctrl + Shift + V	Cmd + Shift + V
Side-by-side Preview	Ctrl + K V	Cmd + K V
Toggle Preview Lock	Ctrl + Shift + L	Cmd + Shift + L
Insert Link	[text](url)	same
Insert Code Block	<code>```lang</code>	same
Task List Checkbox	- [ ] task	same
⚙️ VS Code Markdown Tips

✅ Enable live preview:
Press Ctrl + Shift + V (or Cmd + Shift + V on Mac).

✅ Split editor preview:
Press Ctrl + K V.

✅ Auto-format lists:
Turn on editor.formatOnType in your VS Code settings.

✅ Install extensions:

Markdown All in One — shortcuts & auto TOC

Markdown Preview Enhanced — diagrams, math, mermaid charts

MarkdownLint — ensures clean formatting

🧮 Extras

Footnotes:

This is a statement.[^1]
[^1]: This is the footnote text.


Tables of Contents (with extension):

[TOC]


Mermaid Diagrams (with extension):

<pre> ```mermaid graph TD; A-->B; B-->C; ``` </pre>