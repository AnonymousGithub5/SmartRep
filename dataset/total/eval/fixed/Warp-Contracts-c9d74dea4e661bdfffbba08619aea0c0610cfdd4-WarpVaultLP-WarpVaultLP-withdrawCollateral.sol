function FunctionDefinition_0 ( uint256 _amount ) public { uint256 amount ; uint256 VariableDeclaration_0 = Identifier_0 . MemberAccess_0 ( msg . sender , address ( Identifier_1 ) ) ; if ( _amount == 0 ) { amount = Identifier_2 ; } else { amount = _amount ; } require ( Identifier_3 >= amount , stringLiteral_0 ) ; require ( Identifier_4 [ msg . sender ] >= amount , stringLiteral_1 ) ; Identifier_5 [ msg . sender ] = Identifier_6 [ msg . sender ] . sub ( amount ) ; Identifier_7 . transfer ( msg . sender , amount ) ; emit Identifier_8 ( msg . sender , amount ) ; }