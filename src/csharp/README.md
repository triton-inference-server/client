# Example C# Client


## Prerequsites
.NET 5.0

## Referencing the proto file in csproj
Running in windows command prompt:

Clone the [triton-inference-server/common](https://github.com/triton-inference-server/common/)
repository:

```
C:\src>git clone https://github.com/triton-inference-server/common.git -b <common-repo-branch> common-repo
```

\<common-repo-branch\> should be the version of the Triton server that you
intend to use (e.g. r21.05).

Copy __*.proto__ files to proto folder 

```
C:\src\client\src\csharp>copy ..\..\..\common-repo\protobuf\*.proto proto\.
```

These proto files are referenced in the .csproj file and the client stubs are automatically generated.


Build the sample project
```
c:\src\client\src\csharp>cd examples\simplegrpcclient
c:\src\client\src\csharp\examples\simplegrpcclient>dotnet build
```

To run the Triton server, follow [this guide](https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md) and it works with WSL2 (CPU mode)

To run the client
```
c:\src\client\src\csharp\examples\simplegrpcclient>dotnet bin\Debug\net5.0\simplegrpcclient.dll
```
