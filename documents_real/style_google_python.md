<!-- Source: https://raw.githubusercontent.com/google/styleguide/gh-pages/pyguide.md -->
<!-- Domain: raw.githubusercontent.com -->

<!--
AUTHORS:
Prefer only GitHub-flavored Markdown in external text.
See README.md for details.
-->

# Google Python Style Guide

<!-- markdown="1" is required for GitHub Pages to render the TOC properly. -->

<details markdown="1">
  <summary>Table of Contents</summary>

-   [1 Background](#s1-background)
-   [2 Python Language Rules](#s2-python-language-rules)
    *   [2.1 Lint](#s2.1-lint)
    *   [2.2 Imports](#s2.2-imports)
    *   [2.3 Packages](#s2.3-packages)
    *   [2.4 Exceptions](#s2.4-exceptions)
    *   [2.5 Mutable Global State](#s2.5-global-variables)
    *   [2.6 Nested/Local/Inner Classes and Functions](#s2.6-nested)
    *   [2.7 Comprehensions & Generator Expressions](#s2.7-comprehensions)
    *   [2.8 Default Iterators and Operators](#s2.8-default-iterators-and-operators)
    *   [2.9 Generators](#s2.9-generators)
    *   [2.10 Lambda Functions](#s2.10-lambda-functions)
    *   [2.11 Conditional Expressions](#s2.11-conditional-expressions)
    *   [2.12 Default Argument Values](#s2.12-default-argument-values)
    *   [2.13 Properties](#s2.13-properties)
    *   [2.14 True/False Evaluations](#s2.14-truefalse-evaluations)
    *   [2.16 Lexical Scoping](#s2.16-lexical-scoping)
    *   [2.17 Function and Method Decorators](#s2.17-function-and-method-decorators)
    *   [2.18 Threading](#s2.18-threading)
    *   [2.19 Power Features](#s2.19-power-features)
    *   [2.20 Modern Python: from \_\_future\_\_ imports](#s2.20-modern-python)
    *   [2.21 Type Annotated Code](#s2.21-type-annotated-code)
-   [3 Python Style Rules](#s3-python-style-rules)
    *   [3.1 Semicolons](#s3.1-semicolons)
    *   [3.2 Line length](#s3.2-line-length)
    *   [3.3 Parentheses](#s3.3-parentheses)
    *   [3.4 Indentation](#s3.4-indentation)
        +   [3.4.1 Trailing commas in sequences of items?](#s3.4.1-trailing-commas)
    *   [3.5 Blank Lines](#s3.5-blank-lines)
    *   [3.6 Whitespace](#s3.6-whitespace)
    *   [3.7 Shebang Line](#s3.7-shebang-line)
    *   [3.8 Comments and Docstrings](#s3.8-comments-and-docstrings)
        +   [3.8.1 Docstrings](#s3.8.1-comments-in-doc-strings)
        +   [3.8.2 Modules](#s3.8.2-comments-in-modules)
        +   [3.8.2.1 Test modules](#s3.8.2.1-test-modules)
        +   [3.8.3 Functions and Methods](#s3.8.3-functions-and-methods)
        +   [3.8.3.1 Overridden Methods](#s3.8.3.1-overridden-methods)
        +   [3.8.4 Classes](#s3.8.4-comments-in-classes)
        +   [3.8.5 Block and Inline Comments](#s3.8.5-block-and-inline-comments)
        +   [3.8.6 Punctuation, Spelling, and Grammar](#s3.8.6-punctuation-spelling-and-grammar)
    *   [3.10 Strings](#s3.10-strings)
        +   [3.10.1 Logging](#s3.10.1-logging)
        +   [3.10.2 Error Messages](#s3.10.2-error-messages)
    *   [3.11 Files, Sockets, and similar Stateful Resources](#s3.11-files-sockets-closeables)
    *   [3.12 TODO Comments](#s3.12-todo-comments)
    *   [3.13 Imports formatting](#s3.13-imports-formatting)
    *   [3.14 Statements](#s3.14-statements)
    *   [3.15 Accessors](#s3.15-accessors)
    *   [3.16 Naming](#s3.16-naming)
        +   [3.16.1 Names to Avoid](#s3.16.1-names-to-avoid)
        +   [3.16.2 Naming Conventions](#s3.16.2-naming-conventions)
        +   [3.16.3 File Naming](#s3.16.3-file-naming)
        +   [3.16.4 Guidelines derived from Guido's Recommendations](#s3.16.4-guidelines-derived-from-guidos-recommendations)
    *   [3.17 Main](#s3.17-main)
    *   [3.18 Function length](#s3.18-function-length)
    *   [3.19 Type Annotations](#s3.19-type-annotations)
        +   [3.19.1 General Rules](#s3.19.1-general-rules)
        +   [3.19.2 Line Breaking](#s3.19.2-line-breaking)
        +   [3.19.3 Forward Declarations](#s3.19.3-forward-declarations)
        +   [3.19.4 Default Values](#s3.19.4-default-values)
        +   [3.19.5 NoneType](#s3.19.5-nonetype)
        +   [3.19.6 Type Aliases](#s3.19.6-type-aliases)
        +   [3.19.7 Ignoring Types](#s3.19.7-ignoring-types)
        +   [3.19.8 Typing Variables](#s3.19.8-typing-variables)
        +   [3.19.9 Tuples vs Lists](#s3.19.9-tuples-vs-lists)
        +   [3.19.10 Type variables](#s3.19.10-typevars)
        +   [3.19.11 String types](#s3.19.11-string-types)
        +   [3.19.12 Imports For Typing](#s3.19.12-imports-for-typing)
        +   [3.19.13 Conditional Imports](#s3.19.13-conditional-imports)
        +   [3.19.14 Circular Dependencies](#s3.19.14-circular-dependencies)
        +   [3.19.15 Generics](#s3.19.15-generics)
        +   [3.19.16 Build Dependencies](#s3.19.16-build-dependencies)
-   [4 Parting Words](#4-parting-words)

</details>

<a id="s1-background"></a>
<a id="1-background"></a>

<a id="background"></a>
## 1 Background 

Python is the main dynamic language used at Google. This style guide is a list
of *dos and don'ts* for Python programs.

To help you format code correctly, we've created a [settings file for Vim](google_python_style.vim). For Emacs, the default settings should be fine.

Many teams use the [Black](https://github.com/psf/black) or [Pyink](https://github.com/google/pyink)
auto-formatter to avoid arguing over formatting.


<a id="s2-python-language-rules"></a>
<a id="2-python-language-rules"></a>

<a id="python-language-rules"></a>
## 2 Python Language Rules 

<a id="s2.1-lint"></a>
<a id="21-lint"></a>

<a id="lint"></a>
### 2.1 Lint 

Run `pylint` over your code using this [pylintrc](https://google.github.io/styleguide/pylintrc).

<a id="s2.1.1-definition"></a>
<a id="211-definition"></a>

<a id="lint-definition"></a>
#### 2.1.1 Definition 

`pylint`
is a tool for finding bugs and style problems in Python source code. It finds
problems that are typically caught by a compiler for less dynamic languages like
C and C++. Because of the dynamic nature of Python, some
warnings may be incorrect; however, spurious warnings should be fairly
infrequent.

<a id="s2.1.2-pros"></a>
<a id="212-pros"></a>

<a id="lint-pros"></a>
#### 2.1.2 Pros 

Catches easy-to-miss errors like typos, using-vars-before-assignment, etc.

<a id="s2.1.3-cons"></a>
<a id="213-cons"></a>

<a id="lint-cons"></a>
#### 2.1.3 Cons 

`pylint`
isn't perfect. To take advantage of it, sometimes we'll need to write around it,
suppress its warnings or fix it.

<a id="s2.1.4-decision"></a>
<a id="214-decision"></a>

<a id="lint-decision"></a>
#### 2.1.4 Decision 

Make sure you run
`pylint`
on your code.


Suppress warnings if they are inappropriate so that other issues are not hidden.
To suppress warnings, you can set a line-level comment:

```python
def do_PUT(self):  # WSGI name, so pylint: disable=invalid-name
  ...
```

`pylint`
warnings are each identified by symbolic name (`empty-docstring`)
Google-specific warnings start with `g-`.

If the reason for the suppression is not clear from the symbolic name, add an
explanation.

Suppressing in this way has the advantage that we can easily search for
suppressions and revisit them.

You can get a list of
`pylint`
warnings by doing:

```shell
pylint --list-msgs
```

To get more information on a particular message, use:

```shell
pylint --help-msg=invalid-name
```

Prefer `pylint: disable` to the deprecated older form `pylint: disable-msg`.

Unused argument warnings can be suppressed by deleting the variables at the
beginning of the function. Always include a comment explaining why you are
deleting it. "Unused." is sufficient. For example:

```python
def viking_cafe_order(spam: str, beans: str, eggs: str | None = None) -> str:
    del beans, eggs  # Unused by vikings.
    return spam + spam + spam
```

Other common forms of suppressing this warning include using '`_`' as the
identifier for the unused argument or prefixing the argument name with
'`unused_`', or assigning them to '`_`'. These forms are allowed but no longer
encouraged. These break callers that pass arguments by name and do not enforce
that the arguments are actually unused.

<a id="s2.2-imports"></a>
<a id="22-imports"></a>

<a id="imports"></a>
### 2.2 Imports 

Use `import` statements for packages and modules only, not for individual types,
classes, or functions.

<a id="s2.2.1-definition"></a>
<a id="221-definition"></a>

<a id="imports-definition"></a>
#### 2.2.1 Definition 

Reusability mechanism for sharing code from one module to another.

<a id="s2.2.2-pros"></a>
<a id="222-pros"></a>

<a id="imports-pros"></a>
#### 2.2.2 Pros 

The namespace management convention is simple. The source of each identifier is
indicated in a consistent way; `x.Obj` says that object `Obj` is defined in
module `x`.

<a id="s2.2.3-cons"></a>
<a id="223-cons"></a>

<a id="imports-cons"></a>
#### 2.2.3 Cons 

Module names can still collide. Some module names are inconveniently long.

<a id="s2.2.4-decision"></a>
<a id="224-decision"></a>

<a id="imports-decision"></a>
#### 2.2.4 Decision 

*   Use `import x` for importing packages and modules.
*   Use `from x import y` where `x` is the package prefix and `y` is the module
    name with no prefix.
*   Use `from x import y as z` in any of the following circumstances:
    -   Two modules named `y` are to be imported.
    -   `y` conflicts with a top-level name defined in the current module.
    -   `y` conflicts with a common parameter name that is part of the public
        API (e.g., `features`).
    -   `y` is an inconveniently long name.
    -   `y` is too generic in the context of your code (e.g., `from
        storage.file_system import options as fs_options`).
*   Use `import y as z` only when `z` is a standard abbreviation (e.g., `import
    numpy as np`).

For example the module `sound.effects.echo` may be imported as follows:

```python
from sound.effects import echo
...
echo.EchoFilter(input, output, delay=0.7, atten=4)
```

Do not use relative names in imports. Even if the module is in the same package,
use the full package name. This helps prevent unintentionally importing a
package twice.

<a id="imports-exemptions"></a>
##### 2.2.4.1 Exemptions 

Exemptions from this rule:

*   Symbols from the following modules are used to support static analysis and
    type checking:
    *   [`typing` module](#typing-imports)
    *   [`collections.abc` module](#typing-imports)
    *   [`typing_extensions` module](https://github.com/python/typing_extensions/blob/main/README.md)
*   Redirects from the
    [six.moves module](https://six.readthedocs.io/#module-six.moves).

<a id="s2.3-packages"></a>
<a id="23-packages"></a>

<a id="packages"></a>
### 2.3 Packages 

Import each module using the full pathname location of the module.

<a id="s2.3.1-pros"></a>
<a id="231-pros"></a>

<a id="packages-pros"></a>
#### 2.3.1 Pros 

Avoids conflicts in module names or incorrect imports due to the module search
path not being what the author expected. Makes it easier to find modules.

<a id="s2.3.2-cons"></a>
<a id="232-cons"></a>

<a id="packages-cons"></a>
#### 2.3.2 Cons 

Makes it harder to deploy code because you have to replicate the package
hierarchy. Not really a problem with modern deployment mechanisms.

<a id="s2.3.3-decision"></a>
<a id="233-decision"></a>

<a id="packages-decision"></a>
#### 2.3.3 Decision 

All new code should import each module by its full package name.

Imports should be as follows:

```python
Yes:
  # Reference absl.flags in code with the complete name (verbose).
  import absl.flags
  from doctor.who import jodie

  _FOO = absl.flags.DEFINE_string(...)
```

```python
Yes:
  # Reference flags in code with just the module name (common).
  from absl import flags
  from doctor.who import jodie

  _FOO = flags.DEFINE_string(...)
```

*(assume this file lives in `doctor/who/` where `jodie.py` also exists)*

```python
No:
  # Unclear what module the author wanted and what will be imported.  The actual
  # import behavior depends on external factors controlling sys.path.
  # Which possible jodie module did the author intend to import?
  import jodie
```

The directory the main binary is located in should not be assumed to be in
`sys.path` despite that happening in some environments. This being the case,
code should assume that `import jodie` refers to a third-party or top-level
package named `jodie`, not a local `jodie.py`.


<a id="s2.4-exceptions"></a>
<a id="24-exceptions"></a>

<a id="exceptions"></a>
### 2.4 Exceptions 

Exceptions are allowed but must be used carefully.

<a id="s2.4.1-definition"></a>
<a id="241-definition"></a>

<a id="exceptions-definition"></a>
#### 2.4.1 Definition 

Exceptions are a means of breaking out of normal control flow to handle errors
or other exceptional conditions.

<a id="s2.4.2-pros"></a>
<a id="242-pros"></a>

<a id="exceptions-pros"></a>
#### 2.4.2 Pros 

The control flow of normal operation code is not cluttered by error-handling
code. It also allows the control flow to skip multiple frames when a certain
condition occurs, e.g., returning from N nested functions in one step instead of
having to plumb error codes through.

<a id="s2.4.3-cons"></a>
<a id="243-cons"></a>

<a id="exceptions-cons"></a>
#### 2.4.3 Cons 

May cause the control flow to be confusing. Easy to miss error cases when making
library calls.

<a id="s2.4.4-decision"></a>
<a id="244-decision"></a>

<a id="exceptions-decision"></a>

[Document truncated for evaluation purposes]