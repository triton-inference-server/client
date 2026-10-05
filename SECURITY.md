<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Security Policy

## Reporting a Vulnerability

Please do **not** report security vulnerabilities through public GitHub
issues, discussions, or pull requests.

To report a potential security vulnerability in any NVIDIA product, use one
of the following channels:

* **NVIDIA Vulnerability Disclosure Program** (preferred):
  [Security Vulnerability Submission Form](https://www.nvidia.com/en-us/security/)
* **Email:** [NVIDIA PSIRT](mailto:psirt@nvidia.com). Please encrypt the
  message with NVIDIA's [public PGP key](https://www.nvidia.com/en-us/security/pgp-key).
* **GitHub Private Vulnerability Reporting:** use the **Security** tab of this
  repository, where enabled.

**OEM partners should contact their NVIDIA Customer Program Manager.**

Please include:

1. Product name and version or branch that contains the vulnerability
2. Type of vulnerability (for example code execution, denial of service,
   buffer overflow, information disclosure)
3. Step-by-step instructions to reproduce the issue
4. Proof-of-concept or exploit code, if available
5. Potential impact, including how an attacker could exploit the issue

NVIDIA PSIRT acknowledges reports, assesses severity, coordinates a fix and
disclosure timeline with the reporter, and publishes security bulletins at
<https://www.nvidia.com/en-us/security/>.

## Security Architecture and Context

**Project:** Triton Inference Server client libraries and examples.

**Software classification:** SDK / Library. This repository provides client
libraries that run inside the application of the person using them. It is not
a network service.

**Components:**

* C++ client library (`src/c++/library`): HTTP/REST client built on libcurl
  (`http_client.cc`) and gRPC client built on gRPC C++ (`grpc_client.cc`),
  plus POSIX shared-memory helpers (`shm_utils.cc`) and optional CUDA shared
  memory support.
* Python client packages (`src/python/library/tritonclient`): HTTP, gRPC and
  asyncio clients, tensor serialization helpers (`utils`), and system and CUDA
  shared-memory modules.
* Java HTTP client (`src/java`), Rust client (`src/rust/triton-client`), and
  generated gRPC bindings for Go, Java and JavaScript (`src/grpc_generated`).
* Perf Analyzer packaging: `src/c++/perf_analyzer` and the
  `TRITON_PACKAGE_PERF_ANALYZER` build option (default OFF), which includes
  Perf Analyzer in the pip wheel. Perf Analyzer itself is not analyzed in this
  document.
* Example programs and tests under the `examples` and `tests` directories.

**Primary security responsibility:** correct and safe handling of data
exchanged with a Triton server: establishing the transport securely when the
caller enables it, parsing server responses without memory or resource
corruption, and managing shared-memory regions.

**Key security boundaries and interfaces:**

* Network boundary between the client process and a Triton server (HTTP/REST
  and gRPC).
* Local boundary between the client process and shared-memory regions (POSIX
  shared memory and CUDA IPC handles) that it creates and registers with the
  server.
* Local API boundary between the library and the calling application, which
  supplies URLs, headers, certificates and tensor data.

**Repository Exposure Classification:** Public. This repository is publicly
visible on GitHub.

**Service Exposure Classification:** Not determined (low confidence). The
libraries are deployed by third parties, so exposure depends on the
application and server they connect to.

## Threat Model

1. **Eavesdropping or tampering on unencrypted connections.** Both the HTTP
   and gRPC clients default to plaintext (`ssl=False` in the Python clients;
   the C++ gRPC client uses TLS only when `use_ssl` is set, and the C++ HTTP
   client only when the server URL begins with `https://`). Inference inputs,
   outputs and request headers, including any credentials a caller places in
   them, can be read or modified by a network attacker.

2. **Man-in-the-middle through weakened TLS verification.** The C++ HTTP
   client exposes `HttpSslOptions::verify_peer` and `verify_host` (default 1
   and 2; `http_client.cc`, passed to libcurl), and the Python HTTP client exposes an
   `insecure` flag and caller-supplied `ssl_options`. Callers who disable
   verification allow server impersonation. Omitting an explicit CA bundle
   falls back to the default trust store of libcurl or gRPC rather than
   disabling verification.

3. **Malicious or compromised server response.** Client code decodes
   server-controlled data: JSON and binary-tensor payloads in
   `http_client.cc` and `http/_infer_result.py`, protobuf responses and raw
   output contents in `grpc/_infer_result.py` (`np.frombuffer`), and
   length-prefixed BYTES tensors (`utils.deserialize_bytes_tensor`,
   `struct.unpack_from`). A hostile server could send inconsistent shapes,
   sizes or lengths, or compressed bodies (gzip, deflate) designed to
   exhaust memory or trigger out-of-bounds reads.

4. **Shared-memory abuse on the local host.** The C++ helpers create and map
   POSIX shared-memory objects (`shm_open`, `mmap` in `shm_utils.cc`) and the
   CUDA modules export `cudaIpcMemHandle_t` handles that are base64-encoded
   and registered with the server. Other local users or processes that can
   open a predictable key or obtain a handle may read or alter tensor data.
   Regions that are not unlinked or unregistered remain accessible after the
   client exits.

5. **Unsafe handling of caller-supplied URLs, headers and file paths.** The
   clients forward user-provided headers and query parameters to the server
   and read certificate and key files named by the caller
   (`grpc_client.cc` `ReadFile`). Applications that pass untrusted values
   through these arguments can enable header injection or reads of unintended
   files.

6. **Vulnerable third-party dependencies.** The clients link against libcurl,
   gRPC, protobuf, OpenSSL and, in Python, `numpy`, `grpcio`, `aiohttp` and
   `geventhttpclient`. Known issues in those components affect client users.

## Critical Security Assumptions

* **The network path is trusted unless TLS is enabled by the caller.** The
  libraries do not enable TLS by default. Callers must enable it (an `https://`
  URL for the C++ HTTP client, `use_ssl` for the C++ gRPC client) and, where the
  default trust store is not appropriate, supply the correct CA certificates.
* **Authentication and authorization are performed by the server or proxy.**
  The clients provide no credential management. They only pass headers or
  channel credentials supplied by the caller.
* **The Triton server is trusted to return well-formed responses.** Response
  size and structure are not independently bounded by the client beyond what
  the underlying libraries enforce. Connect only to servers you trust.
* **Local users sharing a host are trusted for shared-memory use.** System
  and CUDA shared memory rely on operating-system permissions and key
  secrecy. Do not use them across a trust boundary between local users.
* **The calling application validates its own inputs.** URLs, header
  values, file paths and tensor shapes passed to the library are assumed to
  come from the trusted application, not directly from end users.
* **Dependencies are kept up to date by the integrator.** Users who build or
  install the clients are responsible for tracking security updates to the
  libraries listed above.
