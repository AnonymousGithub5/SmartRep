function deposit ( address Parameter_0 , uint256 _amount , uint16 ) external { ERC20 token = ERC20 ( Identifier_0 ) ; token . transferFrom ( msg . sender , address ( this ) , _amount ) ; address VariableDeclaration_0 = Identifier_1 [ Identifier_2 ] ; UserDefinedTypeName_0 VariableDeclaration_1 = Identifier_3 ( Identifier_4 ) ; Identifier_5 . mint ( msg . sender , _amount ) ; token . transfer ( Identifier_6 , _amount ) ; }