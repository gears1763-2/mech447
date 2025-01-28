# Python Package: `mech447`

    Anthony Truelove MASc, P.Eng.
    Python Certified Professional Programmer (PCPP1)

***See license terms***

This is a utility package for supporting instruction in MECH 447/542 as
offered at the University of Victoria by Dr. Andrew Rowe.

--------


## Contents

This project is packaged in a manner that is consistent (more or less) with
the structure and layout defined in <https://packaging.python.org/en/latest/tutorials/packaging-projects/>.

In this directory for this project, you should find this README, a LICENSE file,
a bash script for generating the documentation (Linux, Mac), a batch file for
generating the documentation (Windows), a project manifest (toml), and
the following sub-directories:

    docs/       The documentation for the package (start by opening `index.html`
                in the browser of your choice)

    examples/   Some example scripts showing how to import and use the package.

    mech447/    The source code for the package.

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
