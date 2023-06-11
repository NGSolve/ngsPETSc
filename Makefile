lint:
	pylint --disable=C0103,C0321,E1101,R1702,R0914,R0903,R0205,R0912,R0915,I1101 --variable-naming-style=camelCase --class-naming-style=PascalCase --argument-naming-style=camelCase --attr-naming-style=camelCase ngsPETSc
	pylint --disable=C0103,C0321,E1101,R0914,R1702,R0903,R0205,R0912,R0915,I1101 --variable-naming-style=camelCase --class-naming-style=PascalCase --argument-naming-style=camelCase --attr-naming-style=camelCase tests
test:
	pytest tests/ 
