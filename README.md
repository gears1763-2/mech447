# Python Package: `mech447`

    Anthony Truelove MASc, P.Eng.
    Python Certified Professional Programmer (PCPP1)

Copyright 2025 - Anthony Truelove  
SEE LICENSE TERMS [HERE](./LICENSE)

This is a utility package for supporting instruction in MECH 447/542 as
offered at the University of Victoria by Dr. Andrew Rowe.

--------


## Contents

This project is packaged in a manner that is consistent (more or less) with
the structure and layout defined in <https://packaging.python.org/en/latest/tutorials/packaging-projects/>.

In the directory for this project, you should find this README, a LICENSE file,
a bash script for generating the documentation (Linux, Mac), a batch file for
generating the documentation (Windows), a project manifest (`.toml`), and
the following sub-directories:

    docs/       The documentation for the package (start by opening `index.html`
                in the browser of your choice).

    examples/   Some example scripts showing how to import and use the package
                (including an example of doing so within a Jupyter notebook).

    mech447/    The source code for the package.

--------


## Tutorial Videos

To help with getting up to speed with this package, a series of tutorial videos
has been provided:

  * Tutorial 1: <https://youtu.be/ThVEBJksYDU>

  * Tutorial 2: <https://youtu.be/MVXbBcfIJ9Q>

  * Tutorial 3: <https://youtu.be/prYfbBMlF2c>

These videos do assume *some* basic knowledge of Python. That said, there are
many, many free resources for learning Python. For example, check out the
offerings from the Python Institute <https://pythoninstitute.org/study-resources>.

--------


## Autodocumentation

Autodocumentation for this project is achieved using `pdoc` (see
<https://pdoc.dev/>). That said, before making use of either of the `make_docs`
files, you need to make sure that `pdoc` is installed. Fortunately, this is 
easily done using `pip`:

    pip install pdoc

Finally, note the autodocumentation includes generating a `pip` requirements
file within the `mech447/` directory. This depends on the `pipreqs` package, so
be sure to install that as well before making use of either of the `make_docs`
scripts. Again, it's `pip` installable:

    pip install pipreqs

--------
