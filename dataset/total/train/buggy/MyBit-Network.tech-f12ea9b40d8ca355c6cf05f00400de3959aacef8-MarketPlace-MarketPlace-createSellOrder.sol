function FunctionDefinition_0 ( uint256 _amount , uint256 Parameter_0 , address Parameter_1 ) external nonReentrant ModifierInvocation_0 ModifierInvocation_1 ( _amount , _price ) ModifierInvocation_2 ( Identifier_9 ) ModifierInvocation_3 ( Identifier_10 , _amount ) returns ( bool ) { bytes32 id = keccak256 ( Identifier_0 , msg . sender ) ; UserDefinedTypeName_0 VariableDeclaration_0 = Identifier_1 [ id ] ; Identifier_2 . MemberAccess_0 = msg . sender ; Identifier_3 . MemberAccess_1 = Identifier_4 ; Identifier_5 . amount = _amount ; Identifier_6 . MemberAccess_2 = _price ; Identifier_7 ( id , Identifier_8 , msg . sender ) ; return true ; }