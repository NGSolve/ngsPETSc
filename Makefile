lint:
	pylint --disable=C0103,C0321,E1101,R0914,R0903,R0205 --variable-naming-style=camelCase --class-naming-style=PascalCase --argument-naming-style=camelCase --attr-naming-style=camelCase ngsPETSc
	pylint --disable=C0103,C0321,E1101,R0914,R0903,R0205 --variable-naming-style=camelCase --class-naming-style=PascalCase --argument-naming-style=camelCase --attr-naming-style=camelCase tests
test:
	pytest tests/ 
