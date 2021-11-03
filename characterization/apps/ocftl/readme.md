# OCFTL base implementation

This folder includes the modified applications to work with the proposed library. To make the library work we did:
* Modified the source code to add the library object to the application execution.
* Created a new Makefile or CMakeList.txt definition for each option.

## ftlib folder
Contains the standard implementation of OCFTL.

## ftlib-new folder
Contains the new implementation of OCFTL, changing the algorithm.

## ftlib-nng folder
Contains the nng implementation of OCFTL, implementation using the NNG messaging library.

## Observartions:
In each application folder, there will be a specific Makefile (`Makefile.<std|new|nng>`) or a folder containing the cmakelists file (`CMakeLists.txt.<std|new|nng>`). The first case, simple use `make -f Makefile.<std|new|nng>`. For the second, copy the desired cmakelists file from the folder to the root folder of the application as `CMakeLists.txt`. Then proceed as normal to compile the applications.

Important to change the library object instantiation in the application source code. 

If using NNG library, use the instantiation that follows:
```c++
FaultTolerance(int hb_time, int susp_time, int my_rank, int total_size);
``` 

If using MPI library, use the instantiation that follows:
```c++
FaultTolerance(int hb_time, int susp_time, MPI_Comm global_comm);
``` 

Finally, it is important to install NNG if using the NNG version and setting the appropriate install directory in the associated `CMakeLists.txt` files