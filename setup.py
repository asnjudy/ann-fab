#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import os
import platform
import sys

from setuptools import setup, find_packages
from setuptools import Command
from setuptools.command.test import test as TestCommand
from setuptools.command.install import install as InstallCommand

__location__ = os.path.join(
    os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe())))

project_name = 'annfab'


def read_version(package):
    with open(os.path.join(package, '__init__.py'), 'r') as fd:
        for line in fd:
            if line.startswith('__version__ = '):
                return line.split()[-1].strip().strip("'")


version = read_version(project_name)

py_major_version, py_minor_version, _ = (
    int(v.rstrip('+')) for v in platform.python_version_tuple())


def get_install_requirements(path):
    content = open(os.path.join(__location__, path)).read()
    requires = [req for req in content.split('\\n') if req != '']
    return requires


class PyTest(TestCommand):

    user_options = [('cov-html=', None, 'Generate junit html report')]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.cov = None
        self.pytest_args = ['--cov', project_name, '--cov-report',
                            'term-missing']
        self.cov_html = False

    def finalize_options(self):
        TestCommand.finalize_options(self)
        if self.cov_html:
            self.pytest_args.extend(['--cov-report', 'html'])

    def run_tests(self):
        self.run_command('build_proto')

        import pytest

        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


class CMakeBuild(Command):
    description = 'Build library using CMake'
    user_options = []

    def initialize_options(self):
        pass

    def run(self):
        os.chdir(
            os.path.join(__location__, 'cpp'))
        import subprocess
        try:
            subprocess.check_call('./redist.sh')
        except (subprocess.CalledProcessError, OSError):
            print 'ERROR: Error building CPP module'
        os.chdir(__location__)

    def finalize_options(self):
        pass


class Install(InstallCommand):

    def do_egg_install(self):
        self.run_command('build_proto')
        InstallCommand.do_egg_install(self)


setup(name=project_name,
      packages=find_packages(),
      version=version,
      description='Approximate Nearest Neighbor: Faster and Better',
      long_description=open('README.md').read(),
      url='https://github.com/zalando/ann-fab',
      author='Zalando SE',
      license='MIT',
      setup_requires=['flake8', 'protobuf-setuptools'],
      install_requires=get_install_requirements('requirements.txt'),
      tests_require=['pytest-cov', 'pytest'],
      cmdclass={'test': PyTest, 'install': Install, 'build_lib': CMakeBuild},
      test_suite='tests',
      include_package_data=True, )
