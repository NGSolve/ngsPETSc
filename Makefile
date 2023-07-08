MPI_EXEC = ${PETSC_DIR}/${PETSC_ARCH}/bin/mpiexec
lint:
	pylint --disable=C0103,C0415,C0321,E1101,E0611,R1736,R0401,R0801,R0902,R1702,R0913,R0914,R0903,R0205,R0912,R0915,I1101,W0201,C0209 --variable-naming-style=camelCase --class-naming-style=PascalCase --argument-naming-style=camelCase --attr-naming-style=camelCase ngsPETSc
	pylint --disable=C0103,C0415,C0321,E1101,E0611,R1736,R0401,R0914,R0801,R0902,R1702,R0913,R0903,R0205,R0912,R0915,I1101,W0201,C0209 --variable-naming-style=camelCase --class-naming-style=PascalCase --argument-naming-style=camelCase --attr-naming-style=camelCase tests
test:
	pytest tests/ 
test_mpi:
	$(MPI_EXEC) --allow-run-as-root -n 2 pytest --with-mpi tests/
doc:
	rm docs/src/notebooks/*.rst
	jupyter nbconvert --to rst docs/src/notebooks/*.ipynb
