function FunctionDefinition_0 ( uint32 _proposalId , address Parameter_0 , string Parameter_1 , bytes Parameter_2 ) external { if ( msg . sender == _memberAddress ) { require ( Identifier_0 ( _proposalId ) ) ; } else require ( Identifier_1 . MemberAccess_0 ( msg . sender ) ) ; require ( ! Identifier_2 ( _proposalId , _memberAddress ) ) ; governanceDat . MemberAccess_1 ( _proposalId , _memberAddress , Identifier_3 ) ; uint VariableDeclaration_0 = governanceDat . MemberAccess_2 ( _proposalId ) ; governanceDat . MemberAccess_3 ( _proposalId , _memberAddress , Identifier_4 - 1 , Identifier_5 , now ) ; }