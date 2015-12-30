# ann-fab
Approximate nearest neighbor: Faster and Better

[![Build Status](https://travis-ci.org/zalando/ann-fab.svg)](https://travis-ci.org/zalando/ann-fab)


ann-fab is an approximate nearest neighbor search built around the NearPy (http://nearpy.io/) framework.
It was started as a project for Zalando's HackWeek #4 in December 2015.

One of the primary motivators for its development was finding (near) duplicates in large
datasets. In this case, a large number of hashes are used, and the matrix-vector product
contributes significantly to the computational requirement.

The extensions to the NearPy framwork aim to address this, but allowing for parallelisation and also GPU acceleration.


# Getting started:

To ensure that the C++ backend is built, run:
```
python setup.py build_lib
```
This should build and copy the file ```_annfab.so``` to the ```annfab``` module folder.


# Running the tests:

```
python setup.py tests
```
