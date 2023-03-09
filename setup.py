import os
import subprocess
import sys
import hiq
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
build_cuda_ext = torch.cuda.is_available()

if '--no_cuda_ext' in sys.argv:
    sys.argv.remove('--no_cuda_ext')
    build_cuda_ext = False


def get_cuda_bare_metal_version(cuda_dir):
    if cuda_dir is None or not os.path.exists(cuda_dir + "/bin/nvcc"):
      return [None]*3
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
      print("*"*40 + "!!!Warning!!!" + "*"*40)
      print(f"CUDA(nvcc) version({bare_metal_major}.{bare_metal_minor}) does not match the version({torch.version.cuda}) used to compile Pytorch binaries.")
      print(f"We strongly recommend you reinstall Pytorch compiled with CUDA version {bare_metal_major}.{bare_metal_minor}.")
      print("In some cases, even a minor-version mismatch will cause subtle error. Pleas refer to: https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.")
      print("*"*90)


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


if not torch.cuda.is_available():
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to
    # query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs
    # (e.g. during an nvidia-docker build command).
    print(
        '\nWarning: Torch did not find available GPUs on this system.\n',
        'If your intention is to cross-compile, this is not an error.\n'
        'By default, Colossal-AI will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n'
        'Volta (compute capability 7.0), Turing (compute capability 7.5),\n'
        'and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n'
        'If you wish to cross-compile for a single specific architecture,\n'
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n')
    if CUDA_HOME is not None and os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        _, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
        if int(bare_metal_major) == 11:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

print("torch.__version__  = {}".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 8):
    raise RuntimeError("Requires Pytorch 1.8 or newer.\n" +
                       "The latest stable release can be obtained from https://pytorch.org/")


cmdclass = {}
ext_modules = []

# Set up macros for forward/backward compatibility hack around
# https://github.com/pytorch/pytorch/commit/4404762d7dd955383acee92e6f06b48144a0742e
# and
# https://github.com/NVIDIA/apex/issues/456
# https://github.com/pytorch/pytorch/commit/eb7b39e02f7d75c26d8a795ea8c7fd911334da7e#diff-4632522f237f1e4e728cb824300403ac
version_dependent_macros = ['-DVERSION_GE_1_1', '-DVERSION_GE_1_3', '-DVERSION_GE_1_5', '-DUSE_C10D_NCCL']

if build_cuda_ext:
    if CUDA_HOME is None:
        raise RuntimeError(
            "--cuda_ext was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If "
            "you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose "
            "names contain 'devel' will provide nvcc.")
    else:
        check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)

        def cuda_ext_helper(name, sources, extra_cuda_flags):
            return CUDAExtension(
                name=name,
                sources=[os.path.join('energonai/kernel/cuda_native/csrc', path) for path in sources],
                include_dirs=[
                    os.path.join(this_dir, 'energonai/kernel/cuda_native/csrc'),
                    #  '/opt/lcsoftware/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.3.0/nccl-2.9.6-1'
                    #  '-ysovaavjkgjez2fwms4dkvatu5yrxbec/include'
                ],
                extra_compile_args={
                    'cxx': ['-O3'] + version_dependent_macros,
                    'nvcc':
                        append_nvcc_threads(['-O3', '--use_fast_math'] + version_dependent_macros + extra_cuda_flags)
                })

        cc_flag = ['-gencode', 'arch=compute_70,code=sm_70']
        _, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
        if int(bare_metal_major) >= 11:
            cc_flag.append('-gencode')
            cc_flag.append('arch=compute_80,code=sm_80')

        extra_cuda_flags = [
            '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__', '-DTHRUST_IGNORE_CUB_VERSION_CHECK'
        ]
        ext_modules.append(
            cuda_ext_helper('energonai_scale_mask', ['scale_mask_softmax_kernel.cu', 'scale_mask_softmax_wrapper.cpp'],
                            extra_cuda_flags + cc_flag))

        ext_modules.append(
            cuda_ext_helper('energonai_layer_norm', ['layer_norm_cuda_kernel.cu', 'layer_norm_cuda.cpp'],
                            extra_cuda_flags + cc_flag))

        ext_modules.append(
            cuda_ext_helper('energonai_transpose_pad',
                            ['transpose_pad_fusion_wrapper.cpp', 'transpose_pad_fusion_kernel.cu'],
                            extra_cuda_flags + cc_flag))

        ext_modules.append(
            cuda_ext_helper('energonai_linear_func', ['linear_wrapper.cpp'],
                            extra_cuda_flags + cc_flag))

        # ext_modules.append(cuda_ext_helper('energonai_nccl',
        #                                    ['get_ncclid.cpp'],
        #                                    extra_cuda_flags + cc_flag))


def get_version():
    with open('version.txt') as f:
        version = f.read().strip()
        if build_cuda_ext:
            torch_version = '.'.join(torch.__version__.split('.')[:2])
            cuda_version = '.'.join(get_cuda_bare_metal_version(CUDA_HOME)[1:])
            version += f'+torch{torch_version}cu{cuda_version}'
        return version

def package_files(ds):
    paths = []
    for d in ds:
        for (path, directories, filenames) in os.walk(d):
            for filename in filenames:
                if '__pycache__' not in str(filename):
                    paths.append(str(os.path.join(path, filename))[len('energonai/'):])
    return paths

extra_files = package_files(['energonai/'])

#print("ext_modules:", ext_modules)
#print("extra_files:", extra_files)
setup(
    name='energonai',
    maintainer='Juncong Moo;Open Source Community;HPCAiTech',
    url='https://github.com/hpcaitech/EnergonAI',
    maintainer_email='juncongmoo@gmail.com',
    version=hiq.read_file('version.txt')[0],
    packages=find_packages(
        exclude=(
            'benchmark',
            'docker',
            'tests',
            'docs',
            'examples',
            'tests',
            'scripts',
            '*.egg-info',
            'dist',
            'build',
        )),
    description='EnergonAI: An Inference System for Large Transformer Models',
    long_description=hiq.read_file('README.md', by_line=False),
    long_description_content_type="text/markdown",
    license='Apache Software License 2.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension} if ext_modules else {},
    install_requires=hiq.read_file('requirements.txt'),
    entry_points={
        'console_scripts': ['energonai=energonai.cli:typer_click_object', ],
    },
    package_data={"energonai": extra_files, "": ['requirements.txt']},
    classifiers=[
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

