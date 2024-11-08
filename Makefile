MPI_EXEC = ${PETSC_DIR}/${PETSC_ARCH}/bin/mpiexec
GITHUB_ACTIONS_FORMATTING=0

ifeq ($(GITHUB_ACTIONS_FORMATTING), 1)
	PYLINT_FORMAT=--msg-template='::error file={path},line={line},col={column},title={msg_id}::{path}:{line}:{column}: {msg_id} {msg}'
else
	PYLINT_FORMAT=
endif

lint:
	pylint ${PYLINT_FORMAT} --disable=C0412,C0103,C0415,C0321,C3001,E0401,E1101,E0611,R1728,R1736,R0401,R0801,R0902,R1702,R0913,R0914,R0903,R0205,R0912,R0915,R0917,I1101,W0102,W0201,W0212,W0406,C0209 --variable-naming-style=camelCase --class-naming-style=PascalCase --argument-naming-style=camelCase --attr-naming-style=camelCase ngsPETSc
	pylint ${PYLINT_FORMAT} --disable=C0412,C0103,C0415,C0321,C3001,E0401,E1101,E0611,R1728,R1736,R0401,R0801,R0902,R1702,R0913,R0914,R0903,R0205,R0912,R0915,R0917,I1101,W0102,W0201,W0212,W0406,C0209 --variable-naming-style=camelCase --class-naming-style=PascalCase --argument-naming-style=camelCase --attr-naming-style=camelCase ngsPETSc/utils

lint_test:
	pylint ${PYLINT_FORMAT} --disable=C0412,C0103,C0415,C0321,E0401,E1101,E0611,R1728,R1736,R0401,R0914,R0801,R0902,R1702,R0913,R0903,R0205,R0912,R0915,I1101,W0201,C0209 --variable-naming-style=camelCase --class-naming-style=PascalCase --argument-naming-style=camelCase --attr-naming-style=camelCase tests

test:
	pytest tests

test_mpi:
	$(MPI_EXEC) --allow-run-as-root -n 2 pytest --with-mpi tests

doc:
	rm docs/src/notebooks/*.rst
	jupyter nbconvert --to rst docs/src/notebooks/*.ipynb

push: lint lint_test test test_mpi
	git gui
	git push

pushf: lint lint_test test test_mpi
	git gui
	git push -f
