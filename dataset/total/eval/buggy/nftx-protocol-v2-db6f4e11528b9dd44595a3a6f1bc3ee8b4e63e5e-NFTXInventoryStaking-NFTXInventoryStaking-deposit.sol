function deposit ( uint256 Parameter_0 , uint256 _amount ) public virtual override { Identifier_0 ( 10 ) ; ( UserDefinedTypeName_0 VariableDeclaration_0 , UserDefinedTypeName_1 VariableDeclaration_1 , uint256 VariableDeclaration_2 ) = Identifier_1 ( Identifier_2 , msg . sender , _amount , Identifier_3 ) ; Identifier_4 . safeTransferFrom ( msg . sender , address ( Identifier_5 ) , _amount ) ; emit Identifier_6 ( Identifier_7 , _amount , Identifier_8 , NumberLiteral_0 , msg . sender ) ; }