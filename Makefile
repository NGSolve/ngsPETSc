MPI_EXEC = ${PETSC_DIR}/${PETSC_ARCH}/bin/mpiexec
lint:
	pylint --disable=C0412,C0103,C0415,C0321,E0401,E1101,E0611,R1728,R1736,R0401,R0801,R0902,R1702,R0913,R0914,R0903,R0205,R0912,R0915,I1101,W0201,C0209 --variable-naming-style=camelCase --class-naming-style=PascalCase --argument-naming-style=camelCase --attr-naming-style=camelCase ngsPETSc
	pylint --disable=C0412,C0103,C0415,C0321,C3001,E0401,E1101,E0611,R1728,R1736,R0401,R0801,R0902,R1702,R0913,R0914,R0903,R0205,R0912,R0915,I1101,W0201,W0406,W0212,C0209 --variable-naming-style=camelCase --class-naming-style=PascalCase --argument-naming-style=camelCase --attr-naming-style=camelCase ngsPETSc/utils
lint_test:
	pylint --disable=C0412,C0103,C0415,C0321,E0401,E1101,E0611,R1728,R1736,R0401,R0914,R0801,R0902,R1702,R0913,R0903,R0205,R0912,R0915,I1101,W0201,C0209 --variable-naming-style=camelCase --class-naming-style=PascalCase --argument-naming-style=camelCase --attr-naming-style=camelCase tests
test:
	pytest tests/test_env.py
	pytest tests/test_vec.py
	pytest tests/test_mat.py
	pytest tests/test_plex.py
	pytest tests/test_ksp.py
	pytest tests/test_pc.py
	pytest tests/test_eps.py
	pytest tests/test_snes.py
	pytest tests/test_fenicsx.py
test_mpi:
	$(MPI_EXEC) --allow-run-as-root -n 2 pytest --with-mpi tests/test_env.py
	$(MPI_EXEC) --allow-run-as-root -n 2 pytest --with-mpi tests/test_vec.py
	$(MPI_EXEC) --allow-run-as-root -n 2 pytest --with-mpi tests/test_mat.py
	$(MPI_EXEC) --allow-run-as-root -n 2 pytest --with-mpi tests/test_plex.py
	$(MPI_EXEC) --allow-run-as-root -n 2 pytest --with-mpi tests/test_ksp.py
	$(MPI_EXEC) --allow-run-as-root -n 2 pytest --with-mpi tests/test_pc.py
	$(MPI_EXEC) --allow-run-as-root -n 2 pytest --with-mpi tests/test_eps.py
	$(MPI_EXEC) --allow-run-as-root -n 2 pytest --with-mpi tests/test_snes.py
	$(MPI_EXEC) --allow-run-as-root -n 2 pytest --with-mpi tests/test_fenicsx.py
doc:
	rm docs/src/notebooks/*.rst
	jupyter nbconvert --to rst docs/src/notebooks/*.ipynb

push: lint lint_test test test_mpi
	git gui
	git push 

pushf: lint lint_test test test_mpi
	git gui
	git push -f
