function FunctionDefinition_0 ( uint256 _tokenId , uint256 _value ) public view override returns ( uint256 ) { address VariableDeclaration_0 = Identifier_0 [ _tokenId ] ; uint256 amount = Identifier_1 . MemberAccess_0 ( Identifier_2 . totalSupply ( Identifier_3 ) , Identifier_4 . MemberAccess_1 ( Identifier_5 ) , Identifier_6 . MemberAccess_2 ( Identifier_7 ) , _value ) ; return amount ; }